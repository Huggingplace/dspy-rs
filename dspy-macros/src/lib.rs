use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, DeriveInput, Data, Fields, Expr, Lit, Meta};

/// Derive macro that turns a struct into a DSPy Signature.
///
/// Fields annotated with `#[input(...)]` become input fields.
/// Fields annotated with `#[output(...)]` become output fields.
/// The struct's doc comment becomes the signature instruction.
///
/// # Example
///
/// ```ignore
/// #[derive(Signature)]
/// /// Given a question, produce a concise answer.
/// struct QA {
///     #[input(desc = "the question to answer")]
///     question: String,
///     #[output(desc = "a concise factual answer")]
///     answer: String,
/// }
/// ```
#[proc_macro_derive(Signature, attributes(input, output))]
pub fn derive_signature(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let name = &input.ident;

    let doc_instruction = extract_doc_comment(&input);

    let fields = match &input.data {
        Data::Struct(data) => match &data.fields {
            Fields::Named(fields) => &fields.named,
            _ => panic!("Signature derive only supports structs with named fields"),
        },
        _ => panic!("Signature derive only supports structs"),
    };

    let mut input_fields = Vec::new();
    let mut output_fields = Vec::new();

    for field in fields {
        let field_name = field.ident.as_ref().unwrap();
        let field_name_str = field_name.to_string();
        let field_ty = &field.ty;

        let mut is_input = false;
        let mut is_output = false;
        let mut desc = String::new();
        let mut prefix = String::new();

        for attr in &field.attrs {
            if attr.path().is_ident("input") {
                is_input = true;
                parse_field_attr(attr, &mut desc, &mut prefix);
            } else if attr.path().is_ident("output") {
                is_output = true;
                parse_field_attr(attr, &mut desc, &mut prefix);
            }
        }

        if !is_input && !is_output {
            panic!(
                "Field `{}` must have either #[input(...)] or #[output(...)] attribute",
                field_name_str
            );
        }

        let field_info = FieldInfo {
            name: field_name_str.clone(),
            ident: field_name.clone(),
            ty: field_ty.clone(),
            desc,
            prefix,
        };

        if is_input {
            input_fields.push(field_info);
        } else {
            output_fields.push(field_info);
        }
    }

    let input_field_names: Vec<_> = input_fields.iter().map(|f| &f.name).collect();
    let input_field_descs: Vec<_> = input_fields.iter().map(|f| &f.desc).collect();
    let input_field_prefixes: Vec<_> = input_fields.iter().map(|f| &f.prefix).collect();
    let input_field_idents: Vec<_> = input_fields.iter().map(|f| &f.ident).collect();
    let input_field_types: Vec<_> = input_fields.iter().map(|f| &f.ty).collect();

    let output_field_names: Vec<_> = output_fields.iter().map(|f| &f.name).collect();
    let output_field_descs: Vec<_> = output_fields.iter().map(|f| &f.desc).collect();
    let output_field_prefixes: Vec<_> = output_fields.iter().map(|f| &f.prefix).collect();
    let output_field_idents: Vec<_> = output_fields.iter().map(|f| &f.ident).collect();
    let output_field_types: Vec<_> = output_fields.iter().map(|f| &f.ty).collect();

    let all_field_idents: Vec<_> = input_field_idents
        .iter()
        .chain(output_field_idents.iter())
        .collect();
    let all_field_names: Vec<_> = input_field_names
        .iter()
        .chain(output_field_names.iter())
        .collect();

    let expanded = quote! {
        impl dspy::signatures::SignatureFields for #name {
            fn instruction() -> &'static str {
                #doc_instruction
            }

            fn input_fields() -> Vec<dspy::signatures::FieldDescriptor> {
                vec![
                    #(
                        dspy::signatures::FieldDescriptor {
                            name: #input_field_names,
                            desc: #input_field_descs,
                            prefix: #input_field_prefixes,
                            type_name: std::any::type_name::<#input_field_types>(),
                        },
                    )*
                ]
            }

            fn output_fields() -> Vec<dspy::signatures::FieldDescriptor> {
                vec![
                    #(
                        dspy::signatures::FieldDescriptor {
                            name: #output_field_names,
                            desc: #output_field_descs,
                            prefix: #output_field_prefixes,
                            type_name: std::any::type_name::<#output_field_types>(),
                        },
                    )*
                ]
            }

            fn signature_name() -> &'static str {
                stringify!(#name)
            }
        }

        impl dspy::signatures::FromExample for #name {
            fn from_example(example: &dspy::primitives::Example) -> Option<Self> {
                Some(Self {
                    #(
                        #all_field_idents: example.get(#all_field_names)
                            .and_then(|v| serde_json::from_value(v.clone()).ok())?,
                    )*
                })
            }

            fn to_example(&self) -> dspy::primitives::Example {
                let mut ex = dspy::primitives::Example::new();
                #(
                    ex.set(
                        #all_field_names,
                        serde_json::to_value(&self.#all_field_idents).unwrap_or_default(),
                    );
                )*
                ex
            }
        }
    };

    TokenStream::from(expanded)
}

struct FieldInfo {
    name: String,
    ident: syn::Ident,
    ty: syn::Type,
    desc: String,
    prefix: String,
}

fn extract_doc_comment(input: &DeriveInput) -> String {
    let mut docs = Vec::new();
    for attr in &input.attrs {
        if attr.path().is_ident("doc") {
            if let Meta::NameValue(nv) = &attr.meta {
                if let Expr::Lit(expr_lit) = &nv.value {
                    if let Lit::Str(s) = &expr_lit.lit {
                        docs.push(s.value().trim().to_string());
                    }
                }
            }
        }
    }
    if docs.is_empty() {
        String::new()
    } else {
        docs.join(" ")
    }
}

fn parse_field_attr(attr: &syn::Attribute, desc: &mut String, prefix: &mut String) {
    let _ = attr.parse_nested_meta(|meta| {
        if meta.path.is_ident("desc") {
            let value = meta.value()?;
            let lit: Lit = value.parse()?;
            if let Lit::Str(s) = lit {
                *desc = s.value();
            }
        } else if meta.path.is_ident("prefix") {
            let value = meta.value()?;
            let lit: Lit = value.parse()?;
            if let Lit::Str(s) = lit {
                *prefix = s.value();
            }
        }
        Ok(())
    });
}
