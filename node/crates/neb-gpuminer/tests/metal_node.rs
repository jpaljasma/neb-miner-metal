//! Offline, on-device acceptance test of the actual CLI and both mainnet datasets.
#![cfg(all(target_os = "macos", feature = "metal"))]

use std::process::{Command, Stdio};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use neb_chain::demo;
use neb_node::params::NetworkParams;
use neb_node::style::{ColorMode, Live, Log};
use neb_node::{Node, NodeConfig};
use neb_pow::shares::load_datasets;

#[test]
#[ignore = "requires a Metal GPU, loopback and NEB_MAINNET_DATASET"]
fn metal_cli_mines_both_slots_and_the_node_accepts_its_shares() {
    let path = std::path::PathBuf::from(std::env::var_os("NEB_MAINNET_DATASET").expect("dataset file"));
    let payout_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../.neb-payout-address");
    let address = std::fs::read_to_string(payout_path).expect("set the local test payout address");
    let payout = neb_wallet::decode_address("neb", address.trim()).expect("invalid test payout address");
    let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap();
    let t = now.as_secs() as u32;
    let mut chain = demo::easy_params(t, demo::PROJECT_LOCK);
    chain.datasets = load_datasets(&path, None).unwrap();
    chain.difficulty.anchor_bits = 0xdd80_0000; // one solution per 4096 attempts on average
    let network = NetworkParams {
        name: "metal-local",
        magic: *b"METL",
        p2p_port: 0,
        rpc_port: 0,
        work_port: 0,
        address_prefix: "neb",
        protocol_version: 3,
        max_peers: 0,
        user_agent: "metal-test",
        dataset_names: &["Kepler", "TESS"],
        genesis: demo::genesis_at(t, chain.difficulty.anchor_bits, payout),
        chain,
    };
    let dir = std::env::temp_dir().join(format!("neb-metal-test-{}-{}", std::process::id(), now.as_nanos()));
    std::fs::create_dir(&dir).unwrap();
    let node = Node::open(NodeConfig { network, data_dir: dir.clone(), verify_pow: true }).unwrap();
    node.set_miner_lock(payout);
    let live = Live::new();
    let log = Arc::new(Log::new(live.clone(), ColorMode::Never));
    let work = node.start_work_service("127.0.0.1:0".parse().unwrap(), log, live).unwrap();
    let mut child = Command::new(env!("CARGO_BIN_EXE_neb-miner-metal"))
        .args(["--node", &work.addr.to_string(), "--dataset"])
        .arg(path)
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let deadline = Instant::now() + Duration::from_secs(90);
    while node.height() < 3 && Instant::now() < deadline && child.try_wait().unwrap().is_none() {
        std::thread::sleep(Duration::from_millis(100));
    }
    let _ = child.kill();
    let stderr = String::from_utf8_lossy(&child.wait_with_output().unwrap().stderr).into_owned();
    let height = node.height();
    let shares = node.block_at(3).map_or(0, |b| b.shares.len());
    let paid = (1..=height).all(|h| {
        let block = node.block_at(h).expect("mined block");
        let reward = &block.txs[0].outputs[0];
        reward.amount > 0 && reward.lock_data == payout
    });
    drop(work);
    drop(node);
    std::fs::remove_dir_all(dir).unwrap();
    assert!(height >= 3 && shares > 0, "height={height}, listed shares={shares}: {stderr}");
    assert!(paid, "test block credited a different mining lock");
    assert!(!stderr.contains("switching to the fallback scanner"), "{stderr}");
}
