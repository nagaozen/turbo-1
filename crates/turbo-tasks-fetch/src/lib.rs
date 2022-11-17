#![feature(min_specialization)]

use anyhow::Result;
use turbo_tasks::primitives::{BytesVc, OptionStringVc, StringVc};

pub fn register() {
    turbo_tasks::register();
    include!(concat!(env!("OUT_DIR"), "/register.rs"));
}

#[derive(Debug)]
#[turbo_tasks::value(shared)]
pub struct HttpResponse {
    pub status: u16,
    pub body: BytesVc,
}

#[turbo_tasks::function]
pub async fn fetch(url: StringVc, user_agent: OptionStringVc) -> Result<HttpResponseVc> {
    let url = url.await?.clone();
    let user_agent = &*user_agent.await?;
    let client = reqwest::Client::new();

    let mut builder = client.get(url);
    if let Some(user_agent) = user_agent {
        builder = builder.header("User-Agent", user_agent);
    }

    let response = builder.send().await?;
    let status = response.status().as_u16();
    let body = response.bytes().await?.to_vec();

    Ok(HttpResponse {
        status,
        body: BytesVc::cell(body),
    }
    .into())
}
