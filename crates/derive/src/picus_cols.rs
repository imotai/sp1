use quote::quote;
use syn::{
    fold::{self, Fold},
    parse_quote, Data, DeriveInput, Fields,
};

/// Folder that replaces a specific type parameter with another type
struct TypeParamReplacer<'a> {
    from: &'a syn::Ident,
    to: syn::Type,
}

impl<'a> Fold for TypeParamReplacer<'a> {
    fn fold_type(&mut self, ty: syn::Type) -> syn::Type {
        match &ty {
            syn::Type::Path(type_path) if type_path.qself.is_none() => {
                // Check if this is our type parameter
                if type_path.path.segments.len() == 1
                    && type_path.path.segments[0].ident == *self.from
                    && type_path.path.segments[0].arguments.is_empty()
                {
                    // Replace with our target type
                    return self.to.clone();
                }
            }
            _ => {}
        }

        // Recursively fold nested types
        fold::fold_type(self, ty)
    }
}

/// Helper function to replace all occurrences of a type parameter with another type
fn replace_type_param(ty: &syn::Type, from: &syn::Ident, to: syn::Type) -> syn::Type {
    let mut replacer = TypeParamReplacer { from, to };
    replacer.fold_type(ty.clone())
}

/// Derive macro that generates a trait implementation to extract Picus annotation information from
/// columns
///
/// Usage:
/// - `#[derive(PicusCols)]` - Generates PicusColumns trait implementation
/// - Fields can have `#[picus(input)]`, `#[picus(output)]`, `#[picus(selector)]`, or
///   `#[picus(nested)]`
/// - Fields with `#[picus(nested)]` are assumed to implement PicusColumns trait
pub fn picus_cols_derive(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let ast = syn::parse_macro_input!(input as DeriveInput);
    let name = &ast.ident;

    // Extract and validate type parameters
    let type_params: Vec<_> = ast
        .generics
        .params
        .iter()
        .filter_map(|param| match param {
            syn::GenericParam::Type(type_param) => Some(&type_param.ident),
            _ => None,
        })
        .collect();

    if type_params.is_empty() {
        panic!("PicusCols requires at least one type parameter.");
    }

    let first_type_param = type_params[0];
    let other_type_params = &type_params[1..];

    let (impl_generics, type_generics, where_clause) = ast.generics.split_for_impl();

    // Process struct fields
    let fields_data = match &ast.data {
        Data::Struct(data_struct) => match &data_struct.fields {
            Fields::Named(fields_named) => &fields_named.named,
            _ => panic!("Only named fields are supported"),
        },
        _ => panic!("PicusCols can only be derived for structs"),
    };

    // Generate field processing code
    let mut field_processors = Vec::new();

    for field in fields_data {
        let field_name = field.ident.as_ref().unwrap();
        let field_name_str = field_name.to_string();
        let field_type = &field.ty;

        // Check if field has #[picus(...)] attributes
        let mut has_picus_input = false;
        let mut has_picus_output = false;
        let mut has_picus_selector = false;
        let mut has_picus_nested = false;

        for attr in &field.attrs {
            if attr.path.is_ident("picus") {
                if let Ok(syn::Meta::List(meta_list)) = attr.parse_meta() {
                    for nested in &meta_list.nested {
                        if let syn::NestedMeta::Meta(syn::Meta::Path(path)) = nested {
                            if path.is_ident("input") {
                                has_picus_input = true;
                            } else if path.is_ident("output") {
                                has_picus_output = true;
                            } else if path.is_ident("selector") {
                                has_picus_selector = true;
                            } else if path.is_ident("nested") {
                                has_picus_nested = true;
                            }
                        }
                    }
                }
            }
        }

        // Generate size calculation based on the type
        // Replace only the first type parameter with u8, keep others unchanged
        let field_type_with_u8 = replace_type_param(field_type, first_type_param, parse_quote!(u8));
        let size_calc = quote! { std::mem::size_of::<#field_type_with_u8>() };

        // Build the processor code based on which annotations are present
        let mut processor_parts = vec![];

        if has_picus_nested {
            // For nested fields, use the type with u8 substituted for the type parameter
            // This handles all cases including nested generics like Foo<Bar<T>>

            // For nested fields, call collect_picus_info
            processor_parts.push(quote! {
                let field_prefix = if prefix.is_empty() {
                    #field_name_str.to_string()
                } else {
                    format!("{}.{}", prefix, #field_name_str)
                };

                <#field_type_with_u8 as sp1_hypercube::air::PicusColumns>::collect_picus_info(
                    offset + current_offset,
                    &field_prefix,
                    info
                );
            });
        } else {
            // For non-nested fields, add direct annotations
            if has_picus_input {
                processor_parts.push(quote! {
                    info.input_ranges.push((
                        offset + current_offset,
                        offset + current_offset + size,
                        if prefix.is_empty() { #field_name_str.to_string() } else { format!("{}.{}", prefix, #field_name_str) },
                    ));
                });
            }

            if has_picus_output {
                processor_parts.push(quote! {
                    info.output_ranges.push((
                        offset + current_offset,
                        offset + current_offset + size,
                        if prefix.is_empty() { #field_name_str.to_string() } else { format!("{}.{}", prefix, #field_name_str) },
                    ));
                });
            }

            if has_picus_selector {
                // For selectors, we need to verify at compile time that the size is 1
                processor_parts.push(quote! {
                    const _: () = {
                        const SIZE: usize = #size_calc;
                        assert!(SIZE == 1, concat!("Field '", #field_name_str, "' marked as #[picus(selector)] must have size 1"));
                    };
                    info.selector_indices.push((
                        offset + current_offset,
                        if prefix.is_empty() { #field_name_str.to_string() } else { format!("{}.{}", prefix, #field_name_str) },
                    ));
                });
            }
        }

        let processor = quote! {
            {
                let size = #size_calc;
                let field_name = if prefix.is_empty() {
                    #field_name_str.to_string()
                } else {
                    format!("{}.{}", prefix, #field_name_str)
                };

                // Add to field_map only for top-level fields (when prefix is empty)
                if prefix.is_empty() {
                    info.field_map.push((
                        field_name.clone(),
                        offset + current_offset,
                        offset + current_offset + size,
                    ));
                }

                #(#processor_parts)*
                current_offset += size;
            }
        };

        field_processors.push(processor);
    }

    // Generate the implementation
    // Create a type where the first type parameter is u8 and others remain the same
    let trait_impl_type = if other_type_params.is_empty() {
        quote! { #name<u8> }
    } else {
        quote! { #name<u8, #(#other_type_params),*> }
    };

    // Extract trait bounds from the struct's generics for the other type parameters
    let trait_bounds: Vec<_> = ast.generics.params.iter().skip(1).collect();
    let trait_impl_generics = if trait_bounds.is_empty() {
        quote! {}
    } else {
        quote! { <#(#trait_bounds),*> }
    };

    let expanded = quote! {
        impl #trait_impl_generics sp1_hypercube::air::PicusColumns for #trait_impl_type {
            /// Returns complete Picus annotation information
            fn picus_info() -> sp1_hypercube::air::PicusInfo {
                let mut info = sp1_hypercube::air::PicusInfo::default();
                Self::collect_picus_info(0, "", &mut info);
                info
            }

            /// Collects Picus information with a given offset and prefix
            fn collect_picus_info(offset: usize, prefix: &str, info: &mut sp1_hypercube::air::PicusInfo) {
                let mut current_offset = 0;

                #(#field_processors)*
            }
        }

        impl #impl_generics #name #type_generics #where_clause {
            /// Returns Picus annotation information including input ranges, output ranges, and selector indices
            pub fn picus_info() -> sp1_hypercube::air::PicusInfo {
                <#trait_impl_type as sp1_hypercube::air::PicusColumns>::picus_info()
            }
        }
    };

    proc_macro::TokenStream::from(expanded)
}
