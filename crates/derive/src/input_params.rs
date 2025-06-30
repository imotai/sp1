use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, Data, DeriveInput, GenericParam, TypeParamBound};

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
    let field_entries = match &ast.data {
        Data::Struct(data_struct) => data_struct
            .fields
            .iter()
            .filter_map(|field| {
                let field_name = field.ident.as_ref()?;
                let field_name_str = field_name.to_string();

                Some(quote! {
                    (#field_name_str.to_string(), self.#field_name.into())
                })
            })
            .collect::<Vec<_>>(),
        _ => panic!("InputParams can only be derived for structs"),
    };

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

        if num_params == 1 {
            // Case 1: Single type parameter with SP1AirBuilder constraint
            quote! {
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
            // AirBuilder>::
            let remaining_params = ast.generics.params.iter().skip(1).map(|param| {
                if let GenericParam::Type(type_param) = param {
                    let ident = &type_param.ident;
                    let bounds = &type_param.bounds;

                    // Replace AB :: with <ConstraintCompiler as AirBuilder> :: in the bounds
                    let new_bounds = if let Some(first_param) = first_param_name {
                        let bounds_str = quote! { #bounds }.to_string();
                        // Token streams have spaces, so "AB :: Expr" not "AB::Expr"
                        let ab_pattern = format!("{first_param} ::");
                        let replacement =
                            "< sp1_stark :: ir :: ConstraintCompiler as p3_air :: AirBuilder > ::";
                        let new_bounds_str = bounds_str.replace(&ab_pattern, replacement);

                        syn::parse_str::<syn::TypeParam>(&format!("{ident}: {new_bounds_str}"))
                            .unwrap_or_else(|_| type_param.clone())
                    } else {
                        type_param.clone()
                    };

                    quote! { #new_bounds }
                } else {
                    quote! { #param }
                }
            });

            let type_args = ast.generics.params.iter().skip(1).filter_map(|param| {
                if let GenericParam::Type(type_param) = param {
                    let ident = &type_param.ident;
                    Some(quote! { #ident })
                } else {
                    None
                }
            });

            quote! {
                impl<#(#remaining_params),*> #name<sp1_stark::ir::ConstraintCompiler, #(#type_args),*> {
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
