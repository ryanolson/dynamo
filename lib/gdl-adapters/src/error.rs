// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Eager errors across the boundary.
//!
//! An `anyhow::Error` from a Dynamo engine or operator enters gdl as a
//! [`gdl::Error`] whose source owns it, and leaves gdl as the same
//! `anyhow::Error`. Two readers depend on that: Migration walks the source
//! chain for a `DynamoError` to decide whether to retry, and the HTTP handler
//! downcasts the owned error to `HttpError` to pick the status code.

use std::error::Error as StdError;
use std::fmt;
use std::sync::Arc;

use dynamo_runtime::error::{DynamoError, ErrorClass};
use dynamo_runtime::protocols::annotated::Annotated;

/// A Dynamo error carried as a [`gdl::Error`]'s source.
///
/// Its own `source` is the error the `anyhow::Error` holds, not that error's
/// source, so a walk of the chain meets every error the `anyhow::Error`
/// would. The `Arc` lets [`exit`] take the error back out of the `gdl::Error`,
/// which shares its source among clones.
struct Entered(Arc<anyhow::Error>);

impl fmt::Debug for Entered {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&*self.0, f)
    }
}

impl fmt::Display for Entered {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&*self.0, f)
    }
}

impl StdError for Entered {
    fn source(&self) -> Option<&(dyn StdError + 'static)> {
        Some((*self.0).as_ref())
    }
}

/// A Dynamo error entering gdl: a [`gdl::Error`] whose source owns `error`.
/// It leaves gdl, through [`PipelineEngine`](crate::PipelineEngine), as
/// `error` itself, so Migration's walk of the chain and the HTTP handler's
/// downcast still find what they look for.
pub fn enter(error: anyhow::Error) -> gdl::Error {
    gdl::Error::node(error.to_string()).with_source(Entered(Arc::new(error)))
}

/// The Dynamo errors `error` stands for: `error` itself, then each `anyhow::Error` inside its
/// chain that entered gdl through [`enter`].
///
/// An operator body inside a gdl pipeline sees a stage's eager error as an `anyhow::Error`
/// holding the `gdl::Error`, not as the `anyhow::Error` the stage raised. A walk of the chain
/// finds the errors that error holds, but not an `anyhow` context, which only
/// `anyhow::Error::downcast_ref` sees. A check that downcasts to a context type tries each.
pub fn entered(error: &anyhow::Error) -> impl Iterator<Item = &anyhow::Error> {
    std::iter::once(error).chain(
        error
            .chain()
            .filter_map(|cause| cause.downcast_ref::<Entered>())
            .map(|entered| &*entered.0),
    )
}

/// A gdl error leaving gdl.
///
/// An error that entered through [`enter`] leaves as the `anyhow::Error` it
/// entered as. When a clone of the `gdl::Error` is still alive, the error
/// cannot be moved out; it leaves wrapped instead, and its chain still holds
/// every error, but a downcast of the result no longer finds them.
///
/// An error raised by gdl itself leaves as a `DynamoError`: `Cancelled` and
/// `DeadlineExceeded` keep their class, and every other kind is `Internal`.
/// The `gdl::Error` is its cause.
pub(crate) fn exit(error: gdl::Error) -> anyhow::Error {
    let entered = error
        .source()
        .and_then(|source| source.downcast_ref::<Entered>())
        .map(|entered| Arc::clone(&entered.0));
    match entered {
        Some(entered) => {
            drop(error);
            Arc::try_unwrap(entered).unwrap_or_else(|shared| anyhow::Error::new(Entered(shared)))
        }
        None => from_gdl(error).into(),
    }
}

/// A gdl error leaving gdl as a stream item: an error item carrying the
/// `DynamoError` in [`exit`]'s result, or one built as `exit` builds it.
pub(crate) fn exit_item<T>(error: gdl::Error) -> Annotated<T> {
    let error = exit(error);
    let error = match error.downcast::<DynamoError>() {
        Ok(error) => error,
        Err(error) => match error.chain().find_map(|e| e.downcast_ref::<DynamoError>()) {
            Some(found) => found.clone(),
            None => DynamoError::builder().message(format!("{error:#}")).build(),
        },
    };
    let mut item = Annotated::from_error(error.to_string());
    item.error = Some(error);
    item
}

fn from_gdl(error: gdl::Error) -> DynamoError {
    let class = match error.kind() {
        gdl::ErrorKind::Cancelled => ErrorClass::Cancelled,
        gdl::ErrorKind::DeadlineExceeded => ErrorClass::DeadlineExceeded,
        _ => ErrorClass::Internal,
    };
    DynamoError::builder()
        .class(class)
        .message(error.message().to_owned())
        .cause(error)
        .build()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, PartialEq)]
    struct Status(u16);

    impl fmt::Display for Status {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "status {}", self.0)
        }
    }

    impl StdError for Status {}

    /// The reason Migration retries on (`migration.rs`, `is_migration_eligible`).
    const WORKER_UNAVAILABLE: &str = "backend.worker_unavailable";

    fn worker_unavailable() -> anyhow::Error {
        DynamoError::builder()
            .class(ErrorClass::WorkerUnavailable)
            .message("prefill routing is still activating")
            .build()
            .into()
    }

    fn find_dynamo<'a>(error: &'a (dyn StdError + 'static)) -> Option<&'a DynamoError> {
        let mut current = Some(error);
        while let Some(error) = current {
            if let Some(found) = error.downcast_ref::<DynamoError>() {
                return Some(found);
            }
            current = error.source();
        }
        None
    }

    #[test]
    fn the_chain_of_an_entered_error_reaches_its_dynamo_error() {
        let entered = enter(worker_unavailable());
        let found = find_dynamo(&entered).expect("a DynamoError in the chain");
        assert_eq!(found.reason().as_str(), WORKER_UNAVAILABLE);

        // Behind anyhow context, too.
        let entered = enter(worker_unavailable().context("dispatch failed"));
        let found = find_dynamo(&entered).expect("a DynamoError behind context");
        assert_eq!(found.reason().as_str(), WORKER_UNAVAILABLE);
    }

    /// An operator body inside a gdl pipeline sees a stage's error as an `anyhow::Error`
    /// holding the `gdl::Error`. An `anyhow` context, such as the router's unknown policy
    /// class refusal, is not on the chain as its own type, so only [`entered`] finds it.
    #[test]
    fn entered_finds_an_anyhow_context_inside_a_gdl_error() {
        #[derive(Debug)]
        struct Refusal;

        impl fmt::Display for Refusal {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str("refused")
            }
        }

        let refused = || anyhow::Error::new(Status(400)).context(Refusal);
        let in_body = anyhow::Error::new(enter(refused()));
        assert!(in_body.downcast_ref::<Refusal>().is_none());
        assert!(entered(&in_body).any(|e| e.downcast_ref::<Refusal>().is_some()));

        // A Dynamo error that never entered gdl is its own first item.
        let error = refused();
        assert!(
            entered(&error)
                .next()
                .is_some_and(|e| e.downcast_ref::<Refusal>().is_some())
        );

        // An error with no context of the type has none.
        let other = anyhow::Error::new(enter(anyhow::Error::new(Status(400))));
        assert!(entered(&other).all(|e| e.downcast_ref::<Refusal>().is_none()));
    }

    #[test]
    fn an_entered_error_leaves_as_itself() {
        let error = anyhow::Error::new(Status(403));
        let head = error.as_ref() as *const (dyn StdError + Send + Sync) as *const ();
        let left = exit(enter(error));
        assert_eq!(
            left.as_ref() as *const (dyn StdError + Send + Sync) as *const (),
            head
        );
        assert_eq!(left.downcast::<Status>().unwrap(), Status(403));
    }

    #[test]
    fn an_entered_error_with_a_live_clone_leaves_wrapped() {
        let entered = enter(anyhow::Error::new(Status(403)));
        let clone = entered.clone();
        let left = exit(entered);
        assert_eq!(left.to_string(), "status 403");
        assert!(left.chain().any(|e| e.downcast_ref::<Status>().is_some()));
        drop(clone);
    }

    #[test]
    fn a_gdl_cancel_leaves_as_a_cancelled_dynamo_error() {
        let left = exit(gdl::Error::cancelled());
        let error = left.downcast::<DynamoError>().unwrap();
        assert_eq!(error.class(), ErrorClass::Cancelled);

        let left = exit(gdl::Error::deadline_exceeded());
        let error = left.downcast::<DynamoError>().unwrap();
        assert_eq!(error.class(), ErrorClass::DeadlineExceeded);

        let left = exit(gdl::Error::routing("no instance"));
        let error = left.downcast::<DynamoError>().unwrap();
        assert_eq!(error.class(), ErrorClass::Internal);
    }

    #[test]
    fn an_error_item_keeps_the_dynamo_error() {
        let item = exit_item::<()>(enter(worker_unavailable().context("dispatch failed")));
        assert!(item.is_error());
        assert_eq!(
            item.error.expect("an error").reason().as_str(),
            WORKER_UNAVAILABLE
        );

        let item = exit_item::<()>(gdl::Error::cancelled());
        assert!(item.is_error());
        assert_eq!(item.error.expect("an error").class(), ErrorClass::Cancelled);
    }
}
