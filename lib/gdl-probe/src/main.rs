//! Scratch probe: cost per request of N pass-through stages, dynamo's `link`
//! chain against gdl's local composition. One configuration per process, so
//! callgrind can take (Ir(2R) - Ir(R)) / R.
//!
//! Usage: gdl-probe <shape> <stages> <items> <requests>
//!   shape: dyn (link chain of Operators), gdl-op (nested OperatorStages),
//!          gdl-fb (one Hop: N Forward nodes, the sink, N Backward nodes)
//!   stages: 0, 1, 5
//! Prints allocations and bytes per measured request.
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

use futures::{StreamExt, stream};

struct Counting;
static ALLOCS: AtomicU64 = AtomicU64::new(0);
static BYTES: AtomicU64 = AtomicU64::new(0);
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        ALLOCS.fetch_add(1, Relaxed);
        BYTES.fetch_add(l.size() as u64, Relaxed);
        unsafe { System.alloc(l) }
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        unsafe { System.dealloc(p, l) }
    }
    unsafe fn realloc(&self, p: *mut u8, l: Layout, n: usize) -> *mut u8 {
        ALLOCS.fetch_add(1, Relaxed);
        BYTES.fetch_add(n as u64, Relaxed);
        unsafe { System.realloc(p, l, n) }
    }
}
#[global_allocator]
static A: Counting = Counting;

static ITEMS: AtomicU64 = AtomicU64::new(1);
fn k() -> u64 {
    ITEMS.load(Relaxed)
}

mod dyn_side {
    use super::*;
    use dynamo_runtime::engine::{AsyncEngineContextProvider, ResponseStream};
    use dynamo_runtime::pipeline::{
        AsyncEngine, Error, ManyOut, Operator, ServiceBackend, ServiceFrontend, SingleIn, Source,
    };

    pub struct Echo;
    #[async_trait::async_trait]
    impl AsyncEngine<SingleIn<u64>, ManyOut<u64>, Error> for Echo {
        async fn generate(&self, req: SingleIn<u64>) -> Result<ManyOut<u64>, Error> {
            let (v, ctx) = req.into_parts();
            let s = stream::iter((0..k()).map(move |i| v + i));
            Ok(ResponseStream::new(Box::pin(s), ctx.context()))
        }
    }

    pub struct Ident;
    #[async_trait::async_trait]
    impl Operator<SingleIn<u64>, ManyOut<u64>, SingleIn<u64>, ManyOut<u64>> for Ident {
        async fn generate(
            &self,
            req: SingleIn<u64>,
            next: Arc<dyn AsyncEngine<SingleIn<u64>, ManyOut<u64>, Error>>,
        ) -> Result<ManyOut<u64>, Error> {
            let s = next.generate(req).await?;
            let ctx = s.context();
            Ok(ResponseStream::new(Box::pin(s.map(|x| x)), ctx))
        }
    }

    pub type Pipe = Arc<ServiceFrontend<SingleIn<u64>, ManyOut<u64>>>;

    macro_rules! chain {
        ($front:ident; $($op:ident),*) => {{
            let tail = $front.clone()$(.link($op.forward_edge()).unwrap())*
                .link(ServiceBackend::from_engine(Arc::new(Echo))).unwrap();
            chain!(@back tail, $front; ; $($op),*)
        }};
        (@back $tail:expr, $front:ident; $($rev:ident),*; $head:ident $(, $rest:ident)*) => {
            chain!(@back $tail, $front; $head $(, $rev)*; $($rest),*)
        };
        (@back $tail:expr, $front:ident; $($rev:ident),*; ) => {{
            $tail$(.link($rev.backward_edge()).unwrap())*.link_terminal($front.clone()).unwrap();
        }};
    }

    pub fn build(n: usize) -> Pipe {
        let front = ServiceFrontend::<SingleIn<u64>, ManyOut<u64>>::new();
        let o = || Arc::new(Ident).into_operator();
        let (a, b, c, d, e) = (o(), o(), o(), o(), o());
        match n {
            0 => chain!(front;),
            1 => chain!(front; a),
            5 => chain!(front; a, b, c, d, e),
            _ => panic!("stages must be 0, 1 or 5"),
        }
        front
    }

    pub async fn once(p: &Pipe, i: u64) -> u64 {
        let s = p.generate(SingleIn::new(i)).await.unwrap();
        s.fold(0u64, |a, x| async move { a + x }).await
    }
}

mod gdl_side {
    use super::*;
    use gdl::{
        Backward, Cx, Forward, Hop, Items, Next, Operator, OperatorStage, Result, Sink, SinkStage,
        items,
    };

    pub struct Echo;
    impl Sink for Echo {
        type Req = u64;
        type Item = u64;
        async fn generate(&self, _cx: Cx, v: u64) -> Result<Items<u64>> {
            Ok(items(stream::iter((0..k()).map(move |i| Ok(v + i)))))
        }
    }

    pub struct Ident;
    impl Operator for Ident {
        type UpReq = u64;
        type UpItem = u64;
        type DownReq = u64;
        type DownItem = u64;
        async fn generate<N: Next<u64, u64>>(
            &self,
            cx: Cx,
            req: u64,
            next: N,
        ) -> Result<Items<u64>> {
            let s = next.generate(cx, req).await?;
            Ok(items(s.map(|x| x)))
        }
    }

    pub struct Fwd;
    impl Forward for Fwd {
        type In = u64;
        type Out = u64;
        async fn forward(&self, _cx: &Cx, req: u64) -> Result<u64> {
            Ok(req)
        }
    }

    pub struct Bwd;
    impl Backward for Bwd {
        type In = u64;
        type Out = u64;
        fn backward(&self, _cx: &Cx, input: Items<u64>) -> Items<u64> {
            items(input.map(|x| x))
        }
    }

    pub async fn once<N: Next<u64, u64>>(g: &N, i: u64) -> u64 {
        let s = g.generate(Cx::new(), i).await.unwrap();
        s.fold(0u64, |a, x| async move { a + x.unwrap() }).await
    }

    pub fn sink() -> Hop<(), SinkStage<Echo>, ()> {
        Hop::new((), SinkStage::new(Arc::new(Echo)), ())
    }
    pub fn wrap<H: Clone>(h: H) -> Hop<(), OperatorStage<Ident, H>, ()> {
        Hop::new((), OperatorStage::new(Arc::new(Ident), h), ())
    }
}

macro_rules! gdl_op {
    (0) => {
        gdl_side::sink()
    };
    (1) => {
        gdl_side::wrap(gdl_op!(0))
    };
    (2) => {
        gdl_side::wrap(gdl_op!(1))
    };
    (3) => {
        gdl_side::wrap(gdl_op!(2))
    };
    (4) => {
        gdl_side::wrap(gdl_op!(3))
    };
    (5) => {
        gdl_side::wrap(gdl_op!(4))
    };
}

async fn run_gdl<N: gdl::Next<u64, u64>>(g: N, requests: u64) -> (u64, u64) {
    for i in 0..50 {
        gdl_side::once(&g, i).await;
    }
    let (a0, b0) = (ALLOCS.load(Relaxed), BYTES.load(Relaxed));
    for i in 0..requests {
        gdl_side::once(&g, i).await;
    }
    (ALLOCS.load(Relaxed) - a0, BYTES.load(Relaxed) - b0)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 5 {
        eprintln!("usage: gdl-probe <dyn|gdl-op|gdl-fb> <0|1|5> <items> <requests>");
        std::process::exit(2);
    }
    let shape = args[1].as_str();
    let stages: usize = args[2].parse().unwrap();
    ITEMS.store(args[3].parse().unwrap(), Relaxed);
    let requests: u64 = args[4].parse().unwrap();
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let (allocs, bytes) = rt.block_on(async {
        use gdl::{Hop, SinkStage, Source};
        use gdl_side::{Bwd, Echo, Fwd};
        match shape {
            "dyn" => {
                let p = dyn_side::build(stages);
                for i in 0..50 {
                    dyn_side::once(&p, i).await;
                }
                let (a0, b0) = (ALLOCS.load(Relaxed), BYTES.load(Relaxed));
                for i in 0..requests {
                    dyn_side::once(&p, i).await;
                }
                (ALLOCS.load(Relaxed) - a0, BYTES.load(Relaxed) - b0)
            }
            "gdl-op" => match stages {
                0 => run_gdl(Source::new(gdl_op!(0)), requests).await,
                1 => run_gdl(Source::new(gdl_op!(1)), requests).await,
                5 => run_gdl(Source::new(gdl_op!(5)), requests).await,
                _ => panic!("stages must be 0, 1 or 5"),
            },
            "gdl-fb" => {
                let s = || SinkStage::new(Arc::new(Echo));
                let f = || Arc::new(Fwd);
                let b = || Arc::new(Bwd);
                match stages {
                    0 => run_gdl(Source::new(Hop::new((), s(), ())), requests).await,
                    1 => run_gdl(Source::new(Hop::new((f(), ()), s(), (b(), ()))), requests).await,
                    5 => {
                        run_gdl(
                            Source::new(Hop::new(
                                (f(), (f(), (f(), (f(), (f(), ()))))),
                                s(),
                                (b(), (b(), (b(), (b(), (b(), ()))))),
                            )),
                            requests,
                        )
                        .await
                    }
                    _ => panic!("stages must be 0, 1 or 5"),
                }
            }
            _ => panic!("shape must be dyn, gdl-op or gdl-fb"),
        }
    });
    println!(
        "{shape} stages={stages} items={} requests={requests} allocs/req={:.2} bytes/req={:.0}",
        k(),
        allocs as f64 / requests as f64,
        bytes as f64 / requests as f64
    );
}
