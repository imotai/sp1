use proc_macro::TokenStream;
use proc_macro2::TokenStream as TokenStream2;
use quote::quote;
use syn::{
    fold::Fold, parse_macro_input, Data, DeriveInput, GenericArgument, GenericParam, Ident,
    PathArguments, PathSegment, Type, TypeParamBound, TypePath,
};

/// Derive macro for generating a `params_vec` function that returns a vector of tuples
/// containing field names and their values with `.into()` called on them.
///
/// # Example
/// ```compile_fail
/// #[derive(InputParams)]
/// pub struct AddOperationInput<AB: SP1AirBuilder> {
///     pub a: Word<AB::Expr>,
///     pub b: Word<AB::Expr>,
///     pub cols: AddOperation<AB::Var>,
///     pub is_real: AB::Expr,
/// }
/// ```
///
/// Will generate:
/// ```compile_fail
/// impl AddOperationInput<ConstraintCompiler> {
///     fn params_vec(
///         &self,
///     ) -> Vec<(
///         String,
///         Shape<
///             ExprRef<<ConstraintCompiler as AirBuilder>::F>,
///             ExprExtRef<<ConstraintCompiler as ExtensionBuilder>::EF>,
///         >,
///     )> {
///         vec![
///             ("a".to_string(), self.a.into()),
///             ("b".to_string(), self.b.into()),
///             ("cols".to_string(), self.cols.into()),
///             ("is_real".to_string(), self.is_real.into()),
///         ]
///     }
/// }
/// ```
pub fn input_params_derive(input: TokenStream) -> TokenStream {
    let ast = parse_macro_input!(input as DeriveInput);
    let name = &ast.ident;

    // Extract fields from the struct
    let (fields, field_entries) = match &ast.data {
        Data::Struct(data_struct) => data_struct
            .fields
            .iter()
            .filter_map(|field| {
                let field_name = field.ident.as_ref()?;
                let field_name_str = field_name.to_string();

                Some((
                    field.clone(),
                    quote! {
                        (#field_name_str.to_string(), self.#field_name.into())
                    },
                ))
            })
            .unzip::<_, _, Vec<_>, Vec<_>>(),
        _ => panic!("InputParams can only be derived for structs"),
    };

    let field_names = fields
        .iter()
        .map(|field| field.ident.clone().expect("Field should be named."))
        .collect::<Vec<_>>();

    // Check if the first type parameter has SP1AirBuilder bound
    let first_param_name = match ast.generics.params.first() {
        Some(GenericParam::Type(ty)) => Some(&ty.ident),
        _ => None,
    };

    let has_sp1_air_builder = ast.generics.params.first().is_some_and(|param| {
        if let GenericParam::Type(type_param) = param {
            type_param.bounds.iter().any(|bound| {
                if let TypeParamBound::Trait(trait_bound) = bound {
                    trait_bound.path.segments.iter().any(|seg| seg.ident == "SP1AirBuilder")
                } else {
                    false
                }
            })
        } else {
            false
        }
    });

    // Generate the implementation
    let expanded = if has_sp1_air_builder {
        let num_params = ast.generics.params.len();

        let mut folder = ReplaceIdentInTy {
            // We know first_param_name is Some because we checked for it above
            target: first_param_name.unwrap().clone(),
            replacement: Ident::new("A", first_param_name.unwrap().span()),
        };

        // Replace all the instances of the first type parameter with `A`
        let field_type_params = fields
            .iter()
            .map(|field| {
                let name = &field.ident;
                let ty_rewritten = folder.fold_type(field.ty.clone());
                quote! { #name: #ty_rewritten }
            })
            .collect::<Vec<_>>();

        if num_params == 1 {
            // Case 1: Single type parameter with SP1AirBuilder constraint
            quote! {
                impl<A: SP1AirBuilder> #name<A> {
                    pub const fn new(#(#field_type_params),*) -> Self {
                        Self {
                            #(#field_names),*
                        }
                    }
                }

                impl #name<sp1_stark::ir::ConstraintCompiler> {
                    fn params_vec(
                        self,
                    ) -> Vec<(
                        String,
                        sp1_stark::ir::Shape<
                            <sp1_stark::ir::ConstraintCompiler as p3_air::AirBuilder>::Expr,
                            <sp1_stark::ir::ConstraintCompiler as p3_air::ExtensionBuilder>::ExprEF,
                        >,
                    )> {
                        vec![
                            #(#field_entries,)*
                        ]
                    }
                }
            }
        } else {
            // Case 2: Multiple type parameters, first one has SP1AirBuilder constraint
            // Extract the remaining type parameters and substitute AB:: with <ConstraintCompiler as
            let remaining_params_with_constraint_compiler = replace_bounds(
                ast.generics.params.iter().skip(1),
                first_param_name.unwrap().clone(),
                "< sp1_stark :: ir :: ConstraintCompiler as p3_air :: AirBuilder >",
            );

            let remaining_params_with_a = replace_bounds(
                ast.generics.params.iter().skip(1),
                first_param_name.unwrap().clone(),
                "A",
            );

            let type_args = ast.generics.params.iter().skip(1).filter_map(|param| {
                if let GenericParam::Type(type_param) = param {
                    let ident = &type_param.ident;
                    Some(quote! { #ident })
                } else {
                    None
                }
            });

            let type_args_clone = type_args.clone();
            quote! {
                impl<A: SP1AirBuilder, #(#remaining_params_with_a),*> #name<A, #(#type_args_clone),*> {
                    pub const fn new(#(#field_type_params),*) -> Self {
                        Self {
                            #(#field_names),*
                        }
                    }
                }

                impl<#(#remaining_params_with_constraint_compiler),*> #name<sp1_stark::ir::ConstraintCompiler, #(#type_args),*> {
                    fn params_vec(
                        self,
                    ) -> Vec<(
                        String,
                        sp1_stark::ir::Shape<
                            <sp1_stark::ir::ConstraintCompiler as p3_air::AirBuilder>::Expr,
                            <sp1_stark::ir::ConstraintCompiler as p3_air::ExtensionBuilder>::ExprEF,
                        >,
                    )> {
                        vec![
                            #(#field_entries,)*
                        ]
                    }
                }
            }
        }
    } else {
        panic!("InputParams requires the first type parameter to have SP1AirBuilder bound");
    };

    TokenStream::from(expanded)
}

fn replace_bounds<'a, I>(bounds: I, target: Ident, replacement: &'a str) -> Vec<TokenStream2>
where
    I: Iterator<Item = &'a GenericParam>,
{
    bounds
        .map(move |bound| {
            if let GenericParam::Type(type_param) = bound {
                let ident = &type_param.ident;
                let bounds = &type_param.bounds;

                let bounds_str: String = quote! { #bounds }.to_string();
                let target_pattern = format!("{target}");
                let new_bounds_str = bounds_str.replace(&target_pattern, replacement);

                let new_bounds =
                    syn::parse_str::<syn::TypeParam>(&format!("{ident}: {new_bounds_str}"))
                        .unwrap_or_else(|_| type_param.clone());

                quote! { #new_bounds }
            } else {
                quote! { #bound }
            }
        })
        .collect()
}

/// Replaces every occurrence of an Ident in a path with the replacement Ident.
struct ReplaceIdentInTy {
    target: Ident,
    replacement: Ident,
}

impl Fold for ReplaceIdentInTy {
    fn fold_type(&mut self, ty: Type) -> Type {
        match ty {
            Type::Path(tp) => Type::Path(self.fold_type_path(tp)),
            // recurse into anything else (`Option<AB::…>`, [Ab::...], etc.)
            _ => syn::fold::fold_type(self, ty),
        }
    }

    fn fold_type_path(&mut self, mut tp: TypePath) -> TypePath {
        // Change the leading segment if it matches `AB`
        if let Some(seg) = tp.path.segments.first_mut() {
            if seg.ident == self.target {
                seg.ident = self.replacement.clone();
            }
        }

        // Keep walking, so we also catch `AB` that appears *inside* generics.
        syn::fold::fold_type_path(self, tp)
    }

    fn fold_path_segment(&mut self, mut seg: PathSegment) -> PathSegment {
        if seg.ident == self.target {
            seg.ident = self.replacement.clone();
        }

        // Arguments like `Vec<AB::Expr>` still need to be visited.
        if let PathArguments::AngleBracketed(ref mut ab) = seg.arguments {
            ab.args = ab
                .args
                .clone()
                .into_iter()
                .map(|arg| match arg {
                    GenericArgument::Type(ty) => GenericArgument::Type(self.fold_type(ty)),
                    _ => arg,
                })
                .collect();
        }

        seg
    }
}
