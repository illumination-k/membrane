use proc_macro::TokenStream;
use quote::{format_ident, quote};
use syn::meta::ParseNestedMeta;
use syn::{ItemFn, LitStr, parse_macro_input};

struct ToolAttrs {
    name: Option<String>,
    description: Option<String>,
}

impl ToolAttrs {
    fn parse(&mut self, meta: ParseNestedMeta) -> syn::Result<()> {
        if meta.path.is_ident("name") {
            let value: LitStr = meta.value()?.parse()?;
            self.name = Some(value.value());
            Ok(())
        } else if meta.path.is_ident("description") {
            let value: LitStr = meta.value()?.parse()?;
            self.description = Some(value.value());
            Ok(())
        } else {
            Err(meta.error("expected `name` or `description`"))
        }
    }
}

/// Attribute macro that generates a `Tool` implementation from an async function.
///
/// # Usage
///
/// ```ignore
/// #[membrane_tool(name = "search", description = "Search the web")]
/// async fn search(input: SearchInput) -> Result<String, membrane_core::error::Error> {
///     Ok(format!("Results for: {}", input.query))
/// }
/// ```
///
/// This generates:
/// - A struct `SearchTool` (PascalCase of function name + "Tool")
/// - `impl Tool for SearchTool` with `definition()` and `execute()`
/// - The input type must implement `DeserializeOwned + JsonSchema`
#[proc_macro_attribute]
pub fn membrane_tool(attr: TokenStream, item: TokenStream) -> TokenStream {
    let func = parse_macro_input!(item as ItemFn);

    let mut tool_attrs = ToolAttrs {
        name: None,
        description: None,
    };
    let attr_parser = syn::meta::parser(|meta| tool_attrs.parse(meta));
    parse_macro_input!(attr with attr_parser);

    let func_name = &func.sig.ident;

    let tool_name = tool_attrs.name.unwrap_or_else(|| func_name.to_string());

    let tool_description = tool_attrs
        .description
        .unwrap_or_else(|| format!("Tool: {}", tool_name));

    // Generate struct name: function_name -> FunctionNameTool
    let struct_name = format_ident!("{}Tool", to_pascal_case(&func_name.to_string()));

    // Extract the input type from the function signature.
    // Expected signature: async fn name(input: InputType) -> Result<String, Error>
    let input_type = match func.sig.inputs.first() {
        Some(syn::FnArg::Typed(pat_type)) => &pat_type.ty,
        _ => {
            return syn::Error::new_spanned(
                &func.sig,
                "membrane_tool function must have exactly one parameter: (input: InputType)",
            )
            .to_compile_error()
            .into();
        }
    };

    let func_body = &func.block;
    let func_asyncness = &func.sig.asyncness;
    let func_output = &func.sig.output;

    // Get the parameter pattern (e.g., `input`)
    let input_pat = match func.sig.inputs.first() {
        Some(syn::FnArg::Typed(pat_type)) => &pat_type.pat,
        _ => unreachable!(),
    };

    let output = quote! {
        pub struct #struct_name;

        // Keep the original function available
        #func_asyncness fn #func_name(#input_pat: #input_type) #func_output
        #func_body

        impl membrane_core::tool::Tool for #struct_name {
            fn definition(&self) -> membrane_core::tool::ToolDefinition {
                let schema = schemars::schema_for!(#input_type);
                let input_schema = serde_json::to_value(schema)
                    .expect("Failed to serialize JSON Schema");
                membrane_core::tool::ToolDefinition {
                    name: #tool_name.to_string(),
                    description: #tool_description.to_string(),
                    input_schema,
                }
            }

            fn execute(
                &self,
                input: serde_json::Value,
            ) -> ::std::pin::Pin<::std::boxed::Box<dyn ::std::future::Future<Output = ::std::result::Result<String, membrane_core::error::Error>> + Send + '_>> {
                ::std::boxed::Box::pin(async move {
                    let #input_pat: #input_type = serde_json::from_value(input)?;
                    #func_name(#input_pat).await
                })
            }
        }
    };

    output.into()
}

fn to_pascal_case(s: &str) -> String {
    s.split('_')
        .map(|word| {
            let mut chars = word.chars();
            match chars.next() {
                None => String::new(),
                Some(c) => c.to_uppercase().to_string() + &chars.collect::<String>(),
            }
        })
        .collect()
}
