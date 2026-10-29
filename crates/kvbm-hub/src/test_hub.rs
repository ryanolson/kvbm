//! An in-process hub for extension unit tests: a real velo-hub registration
//! runs each extension's `prepare` and `commit`.

use std::net::Ipv4Addr;

use velo_ext::{InstanceId, PeerInfo, WorkerAddress};
use velo_hub::client::{ClientError, Extensions, HubClient};

use crate::protocol::{Feature, KVBM_EXTENSION, RuntimeConfigSummary};
use crate::server::{HubServer, HubServerBuilder};

/// The block size every test peer declares. A test hub whose extension
/// requires a block size uses this one.
pub const BLOCK_SIZE: usize = 4;

pub async fn serve(builder: HubServerBuilder) -> HubServer {
    builder
        .bind_addr(Ipv4Addr::LOCALHOST.into())
        .discovery_port(0)
        .control_port(0)
        .serve()
        .await
        .expect("test hub serves")
}

/// One client holds one registration, so each peer gets its own client.
pub fn client(server: &HubServer) -> HubClient {
    HubClient::builder()
        .host("127.0.0.1")
        .public_port(server.discovery_addr().port())
        .control_port(server.control_addr().port())
        .build()
        .expect("test hub client builds")
}

pub fn peer() -> PeerInfo {
    peer_for(InstanceId::new_v4())
}

/// A peer with no velo address, for a hub that runs without a transport.
pub fn peer_for(instance: InstanceId) -> PeerInfo {
    PeerInfo::new(instance, WorkerAddress::from_encoded(b"test".to_vec()))
}

/// Registers `instance` with one feature. The registration lives as long as
/// the returned client.
pub async fn register_as(
    server: &HubServer,
    instance: InstanceId,
    feature: &Feature,
) -> Result<HubClient, ClientError> {
    let client = client(server);
    register(&client, peer_for(instance), std::slice::from_ref(feature)).await?;
    Ok(client)
}

/// The registration extensions a KVBM client sends for `features`.
pub fn extensions(features: &[Feature], runtime: Option<RuntimeConfigSummary>) -> Extensions {
    let mut extensions = Extensions::new();
    if let Some(runtime) = runtime {
        extensions.insert(
            KVBM_EXTENSION.to_string(),
            serde_json::to_value(runtime).expect("runtime summary encodes"),
        );
    }
    for feature in features {
        extensions.insert(
            feature.key().as_str().to_string(),
            crate::client::feature_payload(feature).expect("feature payload encodes"),
        );
    }
    extensions
}

/// Registers `peer` with `features` and a runtime summary of [`BLOCK_SIZE`].
pub async fn register(
    client: &HubClient,
    peer: PeerInfo,
    features: &[Feature],
) -> Result<(), ClientError> {
    let runtime = RuntimeConfigSummary {
        block_size: Some(BLOCK_SIZE),
        ..Default::default()
    };
    client
        .register_peer(peer, extensions(features, Some(runtime)))
        .await
        .map(|_| ())
}

/// One registration attempt through a real client: the hub's answer, and the
/// client that holds the registration when it succeeded.
pub struct Attempt {
    status: reqwest::StatusCode,
    body: serde_json::Value,
    /// The client that sent the registration.
    pub client: HubClient,
}

impl Attempt {
    pub fn status(&self) -> reqwest::StatusCode {
        self.status
    }

    /// The hub's error message, or an empty string on success.
    pub async fn text(&self) -> serde_json::Result<String> {
        Ok(self.body["message"]
            .as_str()
            .unwrap_or_default()
            .to_string())
    }

    /// The hub's error body (`code` and `message`).
    pub async fn json<T: serde::de::DeserializeOwned>(&self) -> serde_json::Result<T> {
        serde_json::from_value(self.body.clone())
    }
}

/// Registers `peer` with `features` and `runtime`, and returns the hub's
/// answer instead of failing on a rejection.
pub async fn attempt(
    server: &HubServer,
    peer: PeerInfo,
    features: &[Feature],
    runtime: Option<RuntimeConfigSummary>,
) -> Attempt {
    let client = client(server);
    let (status, body) = match client
        .register_peer(peer, extensions(features, runtime))
        .await
    {
        Ok(_) => (reqwest::StatusCode::OK, serde_json::Value::Null),
        Err(ClientError::Hub {
            status,
            code,
            message,
        }) => (
            reqwest::StatusCode::from_u16(status.as_u16()).expect("valid status"),
            serde_json::json!({ "code": code, "message": message }),
        ),
        Err(error) => panic!("registration transport failed: {error}"),
    };
    Attempt {
        status,
        body,
        client,
    }
}
