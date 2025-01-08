use proc_macro::TokenStream;
use quote::{format_ident, quote};
use syn::{
    parse_macro_input, Data, DataEnum, DataStruct, DeriveInput, Fields, Generics, PredicateType,
    TraitBound, TraitBoundModifier, Type, TypeParamBound, WherePredicate,
};

#[proc_macro_derive(CudaSend)]
pub fn derive_cuda_send(input: TokenStream) -> TokenStream {
    let ast = parse_macro_input!(input as DeriveInput);
    impl_cuda_send(&ast).unwrap_or_else(|e| e.to_compile_error()).into()
}

/// Orchestrates creating the impl block.
fn impl_cuda_send(ast: &DeriveInput) -> syn::Result<proc_macro2::TokenStream> {
    // The type name (e.g. "MyStruct" or "MyEnum").
    let name = &ast.ident;

    // If the user wrote something like `struct MyStruct<T>`,
    // we want to propagate those generics and add constraints.
    let mut generics = ast.generics.clone();

    // Gather all field types in a Vec. We'll add `: CudaSend` constraints for them.
    let mut field_types = Vec::new();

    // Examine the AST data (struct, enum, or union).
    match &ast.data {
        Data::Struct(data_struct) => collect_struct_field_types(data_struct, &mut field_types),
        Data::Enum(data_enum) => collect_enum_field_types(data_enum, &mut field_types),
        Data::Union(_) => {
            return Err(syn::Error::new_spanned(
                &ast.ident,
                "CudaSend derive is not supported for unions",
            ))
        }
    }

    // Add `T: CudaSend` constraints for each field type T to the generics' where-clause.
    // (We do this for all field types, though in a real scenario you might
    //  skip known-primitive or known-Send-only types.)
    //
    // This ensures that if a struct has fields A, B, C,
    // we end up with:
    //    where
    //        A: CudaSend,
    //        B: CudaSend,
    //        C: CudaSend,
    add_cudasend_bounds(&mut generics, &field_types)?;

    let (impl_generics, ty_generics, where_clause) = generics.split_for_impl();

    // Build the `change_scope` body:
    //   - For structs, call `change_scope` on each field.
    //   - For enums, match each variant, calling `change_scope` on each field.
    let change_scope_body = match &ast.data {
        Data::Struct(data_struct) => impl_change_scope_struct(data_struct),
        Data::Enum(data_enum) => impl_change_scope_enum(data_enum, name),
        Data::Union(_) => unreachable!(),
    };

    // Generate final code:
    //
    // unsafe impl <generics> Send for Name<generics> where <..> { }
    // unsafe impl <generics> CudaSend for Name<generics> where <..> {
    //     fn change_scope(..) { .. }
    // }
    let expanded = quote! {
        // Mark as Send (assuming it's truly safe).
        unsafe impl #impl_generics ::std::marker::Send for #name #ty_generics #where_clause {}

        // Implement the unsafe trait CudaSend.
        unsafe impl #impl_generics CudaSend for #name #ty_generics #where_clause {
            fn change_scope(&mut self, scope: &TaskScope) {
                #change_scope_body
            }
        }
    };

    Ok(expanded)
}

/// Collect all field types from a struct.
fn collect_struct_field_types(data_struct: &DataStruct, out: &mut Vec<Type>) {
    match &data_struct.fields {
        Fields::Named(fields_named) => {
            for field in &fields_named.named {
                out.push(field.ty.clone());
            }
        }
        Fields::Unnamed(fields_unnamed) => {
            for field in &fields_unnamed.unnamed {
                out.push(field.ty.clone());
            }
        }
        Fields::Unit => {
            // No fields to collect
        }
    }
}

/// Collect all field types from an enum (all variants).
fn collect_enum_field_types(data_enum: &DataEnum, out: &mut Vec<Type>) {
    for variant in &data_enum.variants {
        match &variant.fields {
            Fields::Named(fields_named) => {
                for field in &fields_named.named {
                    out.push(field.ty.clone());
                }
            }
            Fields::Unnamed(fields_unnamed) => {
                for field in &fields_unnamed.unnamed {
                    out.push(field.ty.clone());
                }
            }
            Fields::Unit => {}
        }
    }
}

/// Add `T: CudaSend` bounds to the where-clause for each collected type `T`.
fn add_cudasend_bounds(generics: &mut Generics, field_types: &[Type]) -> syn::Result<()> {
    // We want a `where` clause of the form:
    //
    //     where
    //         Type1: CudaSend,
    //         Type2: CudaSend,
    //         ...
    //
    // We'll push these constraints directly into `generics.where_clause`.
    // In a real scenario, you might want to deduplicate identical types or handle them differently.
    //
    // Also note: if the type is a generic parameter (like `T`),
    // this will effectively yield `T: CudaSend`.
    // If the field type is a concrete type (like `i32`), it yields `i32: CudaSend`.
    // That will likely fail unless there's an impl `CudaSend for i32`.
    //
    // So in practice, you might want to skip known-primitive or known-Send-only types.
    // But here, we do it for all field types to illustrate the concept.

    let where_clause = generics.make_where_clause(); // get or create a `where` clause

    for ty in field_types {
        let pred = type_implements_cudasend(ty);
        where_clause.predicates.push(pred);
    }
    Ok(())
}

/// Create a `WherePredicate` that says `<ty> : CudaSend`.
fn type_implements_cudasend(ty: &Type) -> WherePredicate {
    // This yields: `ty: CudaSend`
    // E.g., if `ty` is `T`, we get `T : CudaSend`.
    // If `ty` is `MyType<X>`, we get `MyType<X> : CudaSend`.
    let trait_bound = TraitBound {
        modifier: TraitBoundModifier::None,
        lifetimes: None,
        path: syn::parse_quote!(CudaSend),
        paren_token: None,
    };

    WherePredicate::Type(PredicateType {
        lifetimes: None,
        // The actual type T.
        bounded_ty: ty.clone(),
        // the colon `:`
        colon_token: <syn::Token![:]>::default(),
        // the bound: `CudaSend`
        bounds: std::iter::once(TypeParamBound::Trait(trait_bound)).collect(),
    })
}

/// Generate the `change_scope` body for a struct.
fn impl_change_scope_struct(data_struct: &DataStruct) -> proc_macro2::TokenStream {
    match &data_struct.fields {
        Fields::Named(fields_named) => {
            // For named fields, destructure as `let Self { ref mut field1, ref mut field2 } = self;`
            let field_idents = fields_named.named.iter().map(|f| f.ident.as_ref().unwrap());
            let calls = field_idents.clone().map(|ident| {
                quote! { #ident.change_scope(scope); }
            });

            quote! {
                let Self { #(ref mut #field_idents),* } = self;
                #( #calls )*
            }
        }
        Fields::Unnamed(fields_unnamed) => {
            // For tuple fields, destructure as `let Self(ref mut field_0, ref mut field_1) = self;`
            let count = fields_unnamed.unnamed.len();
            let field_vars: Vec<_> = (0..count).map(|i| format_ident!("__field{}", i)).collect();
            let destructure = quote! {
                let Self(#(ref mut #field_vars),*) = self;
            };
            let calls = field_vars.iter().map(|var| quote! { #var.change_scope(scope); });

            quote! {
                #destructure
                #( #calls )*
            }
        }
        Fields::Unit => {
            // Nothing to do
            quote! {}
        }
    }
}

/// Generate the `change_scope` body for an enum (match each variant).
fn impl_change_scope_enum(
    data_enum: &DataEnum,
    enum_name: &syn::Ident,
) -> proc_macro2::TokenStream {
    let arms = data_enum.variants.iter().map(|variant| {
        let v_name = &variant.ident;
        match &variant.fields {
            Fields::Named(named) => {
                // destructure => MyEnum::Variant { ref mut x, ref mut y } => { x.change_scope(scope); y.change_scope(scope); }
                let field_idents = named.named.iter().map(|f| f.ident.as_ref().unwrap());
                let destructures = field_idents.clone().map(|id| quote!( ref mut #id ));
                let calls = field_idents.map(|id| quote! { #id.change_scope(scope); });

                quote! {
                    #enum_name::#v_name { #( #destructures ),* } => {
                        #( #calls )*
                    }
                }
            }
            Fields::Unnamed(unnamed) => {
                let len = unnamed.unnamed.len();
                let vars: Vec<_> = (0..len).map(|i| format_ident!("__field{}", i)).collect();
                let destructures = vars.iter().map(|v| quote!( ref mut #v ));
                let calls = vars.iter().map(|v| quote! { #v.change_scope(scope); });

                quote! {
                    #enum_name::#v_name(#( #destructures ),*) => {
                        #( #calls )*
                    }
                }
            }
            Fields::Unit => {
                quote! {
                    #enum_name::#v_name => {}
                }
            }
        }
    });

    quote! {
        match self {
            #( #arms ),*
        }
    }
}
