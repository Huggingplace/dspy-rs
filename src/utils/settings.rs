use std::future::Future;
use std::sync::Arc;

use tokio::sync::RwLock;

use crate::clients::LM;

/// Global configuration for DSPy, analogous to `dspy.settings.configure(...)`.
#[derive(Clone)]
pub struct Settings {
    pub lm: Option<Arc<dyn LM>>,
    pub adapter: Option<String>,
    pub cache: bool,
    pub num_threads: usize,
    pub max_tokens: Option<u32>,
    pub temperature: Option<f64>,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            lm: None,
            adapter: None,
            cache: true,
            num_threads: 4,
            max_tokens: None,
            temperature: None,
        }
    }
}

tokio::task_local! {
    static SCOPED_SETTINGS: Arc<Settings>;
}

static GLOBAL_SETTINGS: std::sync::OnceLock<Arc<RwLock<Settings>>> = std::sync::OnceLock::new();

fn global() -> &'static Arc<RwLock<Settings>> {
    GLOBAL_SETTINGS.get_or_init(|| Arc::new(RwLock::new(Settings::default())))
}

/// Set the global default settings.
///
/// ```ignore
/// dspy::configure(Settings {
///     lm: Some(Arc::new(my_lm)),
///     ..Default::default()
/// });
/// ```
pub fn configure(settings: Settings) {
    let g = global().clone();
    tokio::task::block_in_place(|| {
        let rt = tokio::runtime::Handle::current();
        rt.block_on(async {
            let mut guard = g.write().await;
            *guard = settings;
        });
    });
}

/// Run a future with temporarily overridden settings.
///
/// ```ignore
/// dspy::context(Settings { lm: Some(teacher_lm), ..Default::default() }, async {
///     // teacher model active here
/// }).await;
/// ```
pub async fn context<F, R>(settings: Settings, f: F) -> R
where
    F: Future<Output = R>,
{
    SCOPED_SETTINGS.scope(Arc::new(settings), f).await
}

/// Get the currently active settings (scoped override or global).
pub async fn current_settings() -> Arc<Settings> {
    SCOPED_SETTINGS
        .try_with(|s| s.clone())
        .unwrap_or_else(|_| {
            let g = global().clone();
            Arc::new(
                tokio::task::block_in_place(|| {
                    let rt = tokio::runtime::Handle::current();
                    rt.block_on(async { g.read().await.clone() })
                }),
            )
        })
}

/// Get the currently configured LM, or error if none is set.
pub async fn current_lm() -> anyhow::Result<Arc<dyn LM>> {
    current_settings()
        .await
        .lm
        .clone()
        .ok_or_else(|| anyhow::anyhow!("No LM configured. Call dspy::configure() first."))
}
