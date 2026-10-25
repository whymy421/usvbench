"""G49 (v2.6.5, J29): ONE release pins the D7 + appearing pools of every evaluation seed (CPU only).

Run: python -B scripts/test_g49_seed_pools.py
     USVBENCH_D7_POOL_DIR=<the G49 pool bundle folder> python -B scripts/test_g49_seed_pools.py   (adds the resolve
     tests: every seed x task x vessel through the family's OWN loader against the bundle files)

Seeds: 42 (calibration / diagnostic; the v2.6.4 pins) + the formal seeds 11002 (selection-02) and 41041 41047 41051
41057 (final-01..04), 12 tasks x 2 vessels each.
* Pins (no bundle needed): the seed-42 pins of every family equal the v2.6.4 pins (V264_SEED42, read from the
  v2.6.4 tree when this test was written); every seed / task / vessel has a D7 pin and an appearing pin under the
  family's seed-qualified lookup; a seed without pins resolves to nothing; the station-keeping env switches from
  the v2.6.4 INDEX entry to the entry of cfg.seed (and leaves explicit paths alone); BlueBoat docking D1 is not
  applicable by design (G48 patch).
* Resolve (bundle): the bundle is the G49 bundle (BUNDLE_MANIFEST.txt SHA-256); for every seed, vessel and task the
  family loader (F1 resolve_d7_pool, F2 resolve_pool per dose, F3 load_journal_d7_pool, F4 journal_d7_pool_entry +
  StationD7Pool.load/check_env, F5 load_pool, F6 resolve_pool_source + load_pool, appearing load_pinned_pool) finds
  and verifies the pool; seed 42 resolves to exactly the v2.6.4 file SHA-256 and pool digest; the formal seeds'
  appearing pools carry the current suite version (v7m). The seed-42 APPEARING pins are the v2.6.4 ones, admitted
  under the earlier v6m ladder: same files and digests, but not runnable under v7m (the env refuses the suite
  mismatch; BlueBoat docking s42 offers D1 and is already refused on load after the G48 patch).
"""
from __future__ import annotations

import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
SEEDS = (42, 11002, 41041, 41047, 41051, 41057)
FORMAL_SEEDS = SEEDS[1:]
VESSELS = ("blueboat_v2", "wamv_vrx_v2_opt1")
HAZARD_TASKS = ("reach", "scatter", "ring", "fortress", "forced_crossing", "iceberg", "moving_basin")
D7_TASKS = HAZARD_TASKS + ("path_hazard", "path_following", "station_keeping", "docking", "harbour_mission")
G49_BUNDLE_MANIFEST_SHA256 = "f545aa24144aa038d40c5c7d2107ad7e620ee2bb1711b5dd3662db9488fc6274"
V264_SEED42 = json.loads("""{
 "appearing": {
  "docking": {
   "docking__blueboat_v2__s42.json.xz": [
    "2c0ae49c7c342f4e70570bdeb4db65fd5fe7532cbc9fc1092e2214bd8a0283a3",
    "0fba2772588d62e86871e40187f7dd543ab9d20f49a7b100fda9130718ea2629"
   ],
   "docking__wamv_vrx_v2_opt1__s42.json.xz": [
    "ecf2d40cbebb4fb888fbdfb81e2d604601af2186a6cf77b3e9c375c92da064fb",
    "57ed416a19b3a9d5707abb845ad7844803b69312b0381f7a1a5bb6572eba5c0b"
   ]
  },
  "harbor_mission": {
   "harbour_mission__blueboat_v2__s42.json.xz": [
    "9e05bb209d32a1bf83683de87f54cfe949e6389e6b92f0d801aa88cb830d3e7b",
    "23cf0dee1f3d7a929975e2fb17621d1c004c176420e8d09b9e879d89da2169c4"
   ],
   "harbour_mission__wamv_vrx_v2_opt1__s42.json.xz": [
    "6c986bfe816ad57b4fc49c54b9b95cc6a850cab0cc4142e36fcafdbee1ecede1",
    "4d24491e2e8f3f01067d8b3ef91bf8d565b67640950a0e9f4ddde8b0675e5bd9"
   ]
  },
  "hazard_nav": {
   "forced_crossing__blueboat_v2__s42.json.xz": [
    "6e64e4a5c4ff2605e6fd6012f88a74e8fb7d77e4ce2afe4f96a0871dc1ad8e9a",
    "e416f4de102ae7c2c909e6024f8d86c4618215c0ab62b70a5a5f873d172b2151"
   ],
   "forced_crossing__wamv_vrx_v2_opt1__s42.json.xz": [
    "30fd0a1ea49121fd4a4a66b3650561960042c5e362bfd88966e801ea3783b31a",
    "0e27688305ae5c9878cfee8588f833686d63ad1b822310070f8da8178eea289c"
   ],
   "fortress__blueboat_v2__s42.json.xz": [
    "7cbd8b4498205b3ee0b93f92ba096947a238e3601d9ae74876591b1a93db26e3",
    "1aec08f04a4c2dcb9314f1d5b1dc2f4d8c39d3e6ca624e9370518624c7a55c54"
   ],
   "fortress__wamv_vrx_v2_opt1__s42.json.xz": [
    "c0aa8941ac84dcc68b85af42c1f41df2a08015abd3e5bb12444a2ba332189f03",
    "75cf8e2fb21f0d1ebbc125a73e4843d17a5f417fabdebb7a4c8a1dcbc66f1719"
   ],
   "iceberg__blueboat_v2__s42.json.xz": [
    "e389fdbe098c5eb3d9c4bf2de70cae287f0eec78ea365facb02d233160df2c9b",
    "efd580271264061d59ba200345af4f32ce92ccf249bf85dd17898d916fb22b4d"
   ],
   "iceberg__wamv_vrx_v2_opt1__s42.json.xz": [
    "ab8af60d35f0798cb595ff5ef503d7807f748c041d3a79de29d97f1c3d73f078",
    "935dbdf0134be0c3bf8822d75619ae825bfeccb01787d320d566b8978da6f395"
   ],
   "moving_basin__blueboat_v2__s42.json.xz": [
    "515fc2fe9b1adf184963e40048467bd53d9d0984cf298b208351f807c7bdc07f",
    "2e49a9fadfef2310064a7dbab62b4892b52af13f4f7feef9f7a9003a65e94c24"
   ],
   "moving_basin__wamv_vrx_v2_opt1__s42.json.xz": [
    "1b76b1768b200ede6854a8affa14810aaf3b649e8524db8bc6b013a1dc34b246",
    "cf099d4685186fb1d2fb49ef2bd8470114ef1fdc7bd94589b5176f4a8e2aaf35"
   ],
   "reach__blueboat_v2__s42.json.xz": [
    "b83d54a7d8888b6059b80b8e03ffbba2a804dfc00247f8a6b17b319cb0e145d2",
    "4a8e67c07bcc72e50739b1e0da7ac06e64b0e374725912de759d5a277bfad2a6"
   ],
   "reach__wamv_vrx_v2_opt1__s42.json.xz": [
    "fbf327305ed9e205c9ab8e860c6275fba15ce1bb4725ea3674a318c1414c73d2",
    "e6038881f37af12b8551ad5e9a1b3083bc8f6fc9e288e43b513fba3f72dacd4e"
   ],
   "ring__blueboat_v2__s42.json.xz": [
    "39ab958d9d9e164cdd26c80c2f5dad352880497e5e4f464f9dc7e75b116c3bc1",
    "138c8cdba0da6491313fea4fe992e6aeab43aedb9cfafa2e6b63c85e7daaf8df"
   ],
   "ring__wamv_vrx_v2_opt1__s42.json.xz": [
    "b83e9d8ad617d6aa7e0a6db047106a1d7bb43fc7f22f6903063de066816e38ed",
    "ef6d5a298e4526e674b849c038defb634122b1bb835b4c9f3184d56f8ed4ca22"
   ],
   "scatter__blueboat_v2.json.xz": [
    "a7426370306bbc6fb0223e6fa1882eb26820e6b2bda96d8acb9e3d7b0d49a4df",
    "db4d7890385eb0e858039cdfab02e75fb53c64f6a225c566e94cb6dd6039332a"
   ],
   "scatter__blueboat_v2__s42.json.xz": [
    "943db744c92448b32bc730ecf334bb071a1542546acb52f729ef8f1ee7642f95",
    "ad31739b25f49edc665c206bd40c09978189246473dd7da80c046685a4cbdbf7"
   ],
   "scatter__wamv_vrx_v2_opt1.json.xz": [
    "5a6f2ffb373a18e0e880e581d731814a923562fbed43d358af11763ff6c159bd",
    "78397d61dc1d7c91bb7c5629fc82b90cb0d3d677ba63ad510f2d73d1a6903a0a"
   ],
   "scatter__wamv_vrx_v2_opt1__s42.json.xz": [
    "a2ac93c689d815630c3543e57f8810602c531fc10d6087b27fe3cc53573c74d9",
    "96b63e3090bfdc6b6135ba51f28553a7fd89a7d8913df39294554dab0af8ce17"
   ]
  },
  "path_following": {
   "path_following__blueboat_v2.json.xz": [
    "3a4d3f98605a1038e53d78ba353561f26a1698396b6a7ed5748218bcb66f19d8",
    "f57e5bf9901409e8d50ddb0ed488a9d4ae1fb8d08e849a436a2819d8bf44a984"
   ],
   "path_following__blueboat_v2__s42.json.xz": [
    "0c4a34bcc0612bb1bc9c3a8030e9d337a876237bb39d9d8c40a3d9b683939a55",
    "069d62202d3692337ab8fc1c46b93b6e7189b8188b35de92738398656286c960"
   ],
   "path_following__wamv_vrx_v2_opt1.json.xz": [
    "6cda52ff3e9e8a50bcdf1d1c4082ba05fbc483375b204864b2f5779be50545e9",
    "c7e9de6387bd704d341418fac0842e2f09912f141ff60867ee785df645782b82"
   ],
   "path_following__wamv_vrx_v2_opt1__s42.json.xz": [
    "1ad1e46e8721588ab9476c3a66cff60ddcfc4ad1d0c16f24fcf170fd413dc3af",
    "aa9107610751883898cc194f763d44f35e4bcafa002497db69a2a2f7f06261c5"
   ]
  },
  "path_hazard": {
   "path_hazard__blueboat_v2__s42.json.xz": [
    "478d2210c946601e9ac9e820155fc812ecff3ad41324da5b7ec55220d186748c",
    "5abc1e64014d26494c096cf3b56532ce05709817e842b1b46c2a15e75b6b0af7"
   ],
   "path_hazard__wamv_vrx_v2_opt1__s42.json.xz": [
    "a5cd1ebd369793b613641fa117109f175741b1028381a083bde9e45ccfb456f8",
    "e6e513e1a6dd37dde609d62ca0c71492255a73a2ba03a05bd95b5419f3d5adb6"
   ]
  },
  "station_keeping": {
   "station_keeping__blueboat_v2__s42.json.xz": [
    "b80898876efdf1dcfcd200b625e4bcb37144388fb2c24ec823ac77e443e98d23",
    "081505bbc5d92f6f8f1d421eb5b14f9d85e34a2a27ccd5b783799e5daee84341"
   ],
   "station_keeping__wamv_vrx_v2_opt1__s42.json.xz": [
    "0c8d9526359210e65238c42b1959ad0a33dbf664219eea23b0843c08bf3b0266",
    "51963b7da7565902aeddfb8bfb54f7eb37a78b423df752e4782ea56af7dc74eb"
   ]
  }
 },
 "f1": {
  "forced_crossing__blueboat_v2.json.gz": [
   "8ca014ca49e6f37b79e77278f36c185a7f0c67cdee89b13fa007796370c79f9c",
   "16be1a45572620dbff69af0f67884815775987750a495ca507fa87e99908a0c3"
  ],
  "forced_crossing__wamv_vrx_v2_opt1.json.gz": [
   "15a5791d9cfee3f6cd351124225bd0b01ba5a16f1925cb6239345e55e97e83c8",
   "1e5ea6a272d453098ca766e75ee14c1919cb48920709ffb69108350d7087d179"
  ],
  "fortress__blueboat_v2.json.gz": [
   "d46123a17aab157fb331070f183f0ff14b978f090722110d113920578d199d65",
   "4291f2303f211f31d5dde89d1aefd0ea26ec18c91f863608941260de1b1cbe30"
  ],
  "fortress__wamv_vrx_v2_opt1.json.gz": [
   "b68f2790a933937ec350837ecf20f049e683d6fcc736ecc2bb6c1f5d9adbbbd2",
   "a3d867f7094fb1eff069013d388592e2e6380078eb96084bc95eb5b1125b73e1"
  ],
  "iceberg__blueboat_v2.json.gz": [
   "bd7cda8ba8e7022f29a3c4069c9e5944508c578faa64775f21f50a7eaa61f925",
   "95bf5ff12b1947e4a4346eeab3139765713da519cb98f64e631d0090e2f93a10"
  ],
  "iceberg__wamv_vrx_v2_opt1.json.gz": [
   "cbc21bb3714469b5effcfdc7b749a46204a8854bbd4a99a972f9304c96b23394",
   "040c4ac145cf5a4f1f282453c48f3f937dd23e014c9651d0895b2d8d3075bfc0"
  ],
  "moving_basin__blueboat_v2.json.gz": [
   "e8fd8844d468ee7e877dd69471d2199ace8f14e233c843ef8966ea92fd639e28",
   "331cbd051a1bbbd162603e08c066609cb7d94a6a3ca2dc27ef9c571ebc2604f6"
  ],
  "moving_basin__wamv_vrx_v2_opt1.json.gz": [
   "db3fcdaa92766633f8b35fb8838be61a0442890a98ac90834d3e05fd25abb109",
   "94b451a15742bdd64780b53a29fab7ef73100460591f8987bc3226b60693359d"
  ],
  "reach__blueboat_v2.json.gz": [
   "275f06ff83207f3a2f366e26d0c7a5c2a8f94d95aaf665e9c3e9ff702a22b94f",
   "264e6b0d85fa10ed7ee14020104303b632ca5cac5c752f9d8fea914605a9ab5f"
  ],
  "reach__wamv_vrx_v2_opt1.json.gz": [
   "d1fb96c3fb38d7fe740f1492717d596c8042bd3638bc704249ffb3977d9aaea4",
   "35545fe0ac93bd987d6b15ecf455ff59f5112e81e0365ec7d972a6b5aa941647"
  ],
  "ring__blueboat_v2.json.gz": [
   "c7a2ab509a753f4d82a484de762c705941994ddb21ddee37fd5084b32a08c1df",
   "339d19ab6378ef1d7c2b399b313465c1797e1fc5965e4399c03668552933bc8f"
  ],
  "ring__wamv_vrx_v2_opt1.json.gz": [
   "bfdefd387e32ad71f0979530d305983d50745c3e831eb4acde57d9c2bdc02913",
   "6d958b0e00dd54d4468608a1250ba5eee070db2eac8937682bea9f0a1b2ce91e"
  ],
  "scatter__blueboat_v2.json.gz": [
   "39449a6da48efdc79d86b8dccbb98c79f22178a59e086fd20ef649de998eb453",
   "e154fa79fb9f834716a56a6dd38832eba355e0c2814a05a3f8a3dac8105adb71"
  ],
  "scatter__wamv_vrx_v2_opt1.json.gz": [
   "0378cfeb1100524cb9d4e14e312f87a8cdd489a6687d791ade9ca56b1f90e25e",
   "0aa85818a58fbbd27d7d620ab63f03c757cb71978e0617fdda1da2c58469d696"
  ]
 },
 "f2": {
  "blueboat_v2": {
   "files": {
    "0": [
     "path_hazard_d7_pool_blueboat_v2_D0.json.gz",
     "b3efc1e273c46f2a7a0b3e9f0e2798cd171467104ac3a6a95268b8b85f8c5a6d"
    ],
    "1": [
     "path_hazard_d7_pool_blueboat_v2_D1.json.gz",
     "621e7b91765e509b028d4c9df34e1ef74cdba265fc29016a29b29b3bb1d4cc61"
    ],
    "2": [
     "path_hazard_d7_pool_blueboat_v2_D2.json.gz",
     "559849498a4c345f1e404605293878041a963b47d0e043e5e9f476ba19f2a9b0"
    ],
    "3": [
     "path_hazard_d7_pool_blueboat_v2_D3.json.gz",
     "7f6cca530c022cd5a7bd703253c529280e2933898c47d6b12053e9cec5437e82"
    ],
    "4": [
     "path_hazard_d7_pool_blueboat_v2_D4.json.gz",
     "4ea893e9615852a39993fa9e9124b1caf0f2c8c33f0e4119cca83983e7fa82d1"
    ]
   },
   "pool_digest": "f189699a9812a38dc98a8359960d8a4bf61da70886733464b3e322cbb1375c5e"
  },
  "wamv_vrx_v2_opt1": {
   "files": {
    "0": [
     "path_hazard_d7_pool_wamv_vrx_v2_opt1_D0.json.gz",
     "d397985ec0f2bd02822f863e87da93bd6a00ed718df289c6a48969c0104ce5e0"
    ],
    "1": [
     "path_hazard_d7_pool_wamv_vrx_v2_opt1_D1.json.gz",
     "3c9cc4609cdae727847cda2aae5b533a5a470da64bc36433ab4fab3c54da3bac"
    ],
    "2": [
     "path_hazard_d7_pool_wamv_vrx_v2_opt1_D2.json.gz",
     "f2b751246fdfd710704f32c5e2ec348a6b8b51b527618b5c76060f55dd0e8377"
    ],
    "3": [
     "path_hazard_d7_pool_wamv_vrx_v2_opt1_D3.json.gz",
     "ecde7eb1ab502481b896258699b09de402963532434a8e6270756636bfad534a"
    ],
    "4": [
     "path_hazard_d7_pool_wamv_vrx_v2_opt1_D4.json.gz",
     "6b22e598fa4e3037a4476ce3dc85268811fdbb8b39672b9184a812609af539ae"
    ]
   },
   "pool_digest": "970c06639d89a805d718399c127bf1cf42980bd0ba5321cbacad3df4e724090f"
  }
 },
 "f3": {
  "d7_pool_path_following_blueboat_v2_seed42.d7pool.xz": [
   "401881ce09b7f9327a45a93dff7d46df393278c12f8cfe65a072da17ea8fe44a",
   "b8a2e10bc14f8a78c952882ca05abc03b0fbf73b87c59fcc81c1b45281792a7d",
   "f26e342cfa19337a20102de6974cca2989d2af4fd11fca3759e98edfefa09529"
  ],
  "d7_pool_path_following_wamv_vrx_v2_opt1_seed42.d7pool.xz": [
   "e37e464f7191f3bcf59b2d705a3a8183d58aa56d014feb347059f7b3a518c9df",
   "e6e99548782bbdaf7ca9813ec350f84f112727a20a17ca70a2f88e76cc284aa7",
   "6260b190b1b5d839e952ae93419dd95d6a353e74dbb2d36ed45ae629364a3fd4"
  ]
 },
 "f4": {
  "blueboat_v2/tier2": [
   "station_keeping_blueboat_v2_tier2_seed42.json",
   "c2b0b1d7167363fd038b666e8334010c47b4d2232e6f27e2965dba532c62636e",
   "26b17692d3af8be387054fe328fc9b91744f99d84562af0a6cec21aa270da74b",
   "37f1500bafbb764151cd62c918b649af7c1d1342229e6a9c741ae1a9f61a3856"
  ],
  "wamv_vrx_v2_opt1/tier2": [
   "station_keeping_wamv_vrx_v2_opt1_tier2_seed42.json",
   "dd5df2e735a5652f02a9afe127ca8895e8cd548cf77aec0f63a490818942ecf0",
   "c6bf14415a44c4b2538237070e38ced2d3af8e3be9e552d28300b62e8af37056",
   "0d9aa1433091e25b0d1a4a95488aaaa8417fadae63992b275ddbb952def6fac8"
  ]
 },
 "f5": {
  "docking_blueboat_v2_g2_s42_64x2.json.xz": [
   "a28e85bd2426fa675015642b1ad057921cb37f3137ba7929f5a6959f894e4f4d",
   "65ee10782095d88eb84894f97e8dc208fa6f5bfeb60ced0c6ec1bc7770a526ee"
  ],
  "docking_wamv_vrx_v2_opt1_g2_s42_64x2.json.xz": [
   "334ececea3c23ca808a7bd96ff3cb64601fb2ae42cd2a64ba58d22cf2497bc49",
   "9bf123f8ba6e026dd42ada38030d5b9a11a250cbc995b9b454938f266935abcf"
  ]
 },
 "f6": {
  "harbour_mission_blueboat_v2_seed42.json.gz": [
   "dc908c09d8a271394b0e493678c113801c09f7d20d295553f254dc1c4435b804",
   "2163bca21eac4313cf06a69203d04cbb57cf3e98d899fe34ce3f771c3cf9245e"
  ],
  "harbour_mission_wamv_vrx_v2_opt1_seed42.json.gz": [
   "2ddb6ad2d7c3bddc41e67ed26fb23c8720ea04ea15807a6688872114a863eebd",
   "6169a388ed1c6f8d1f5081cc22946ec3c2fd9c9ed371a60471582adedec77ae2"
  ]
 }
}""")
RESTORE = None
M = SimpleNamespace()


def setUpModule():
    global RESTORE
    sys.path[:0] = [str(ROOT), str(ROOT / "scripts"), str(ROOT / "scripts" / "tests_journal_glue")]
    import glue_fixtures as G
    RESTORE = G.install_isaaclab_stubs()
    if "tasks" not in sys.modules:
        pkg = ModuleType("tasks")
        pkg.__path__ = [str(ROOT / "tasks")]
        sys.modules["tasks"] = pkg
    imp = importlib.import_module
    M.jh = imp("tasks.hazard_nav.journal_hazard_nav")
    M.ph = imp("tasks.path_hazard.path_hazard_journal")
    M.pf = imp("tasks.path_following.journal_path_following")
    M.js = imp("tasks.station_keeping.journal_station")
    M.sk = imp("tasks.station_keeping.station_keeping_env_cfg")
    M.dcfg = imp("tasks.docking.docking_env_cfg")
    M.dpool = imp("tasks.docking.journal_d7_pool")
    M.hpool = imp("tasks.harbor_mission.harbor_d7_pool")
    M.dynamic = imp("tasks._shared.conditions.dynamic")
    M.jconf = imp("tasks._shared.conditions.journal_config")
    M.apool = imp("tasks._shared.conditions.appearing.pool")
    M.aconf = imp("tasks._shared.conditions.appearing.config")
    M.afam = imp("tasks._shared.conditions.appearing.families")
    M.ladder = imp("tasks._shared.conditions.appearing.ladder")


def tearDownModule():
    if RESTORE is not None:
        RESTORE()


def load_json(rel):
    return json.loads((ROOT / rel).read_text(encoding="ascii"))


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def f1_pin(seed, task, vessel):
    pins, grid = M.jh.pinned_pool_table(M.jh.load_pool_manifest(), seed)
    name = M.jh.pool_file_name(task, vessel, seed)
    return name, pins.get(name), grid


def f2_table(seed):
    return M.ph.pinned_seed_table(M.ph.pinned_manifest(), seed)


def f3_pin(seed, vessel):
    manifest, _ = M.pf.read_pool_manifest()
    name = M.pf.pool_file_name(vessel, seed)
    return name, manifest["pools"].get(name)


def f5_pin(seed, vessel):
    name = M.dcfg.journal_d7_pool_name(vessel, 2, seed, 64, 2, True)
    return name, M.dcfg.JOURNAL_D7_POOLS.get(name)


def f6_pin(seed, vessel):
    manifest = load_json("tasks/harbor_mission/d7_pools/POOL_MANIFEST.json")
    for entry in manifest["pools"]:
        if (entry["vessel"], entry["seed"], entry["tier"]) == (vessel, seed, 2):
            return entry
    return None


def appearing_pin(task, vessel, seed):
    family = M.afam.FAMILY_OF_TASK[task]
    manifest, _ = M.apool.read_manifest(M.apool.family_pool_dir(family))
    name = M.apool.pool_file_name(task, vessel, seed)
    return family, name, manifest["pools"].get(name)


class Pins(unittest.TestCase):
    def test_seed42_pins_equal_v264(self):
        f1 = M.jh.load_pool_manifest()
        self.assertEqual(f1["grid"]["seed"], M.jh.LEGACY_POOL_SEED)
        self.assertEqual({n: [e["sha256"], e["pool_digest"]] for n, e in f1["pools"].items()}, V264_SEED42["f1"])
        f2 = M.ph.pinned_manifest()
        self.assertEqual(f2["eval_seed"], 42)
        self.assertEqual({v: {"pool_digest": e["pool_digest"],
                              "files": {d: [f["file"], f["sha256"]] for d, f in e["files"].items()}}
                          for v, e in f2["vessels"].items()}, V264_SEED42["f2"])
        f3, _ = M.pf.read_pool_manifest()
        self.assertEqual({n: [e["file_sha256"], e["container_sha256"], e["pool_digest"]]
                          for n, e in f3["pools"].items() if e["seed"] == 42}, V264_SEED42["f3"])
        f4 = load_json("tasks/station_keeping/d7_pools_approach/INDEX.json")
        self.assertEqual({k: [e["file"], e["sha256"], e["witness_sha256"], e["pool_digest"]]
                          for k, e in f4["pools"].items() if e["seed"] == 42}, V264_SEED42["f4"])
        f5 = load_json("tasks/docking/journal_pools/POOL_MANIFEST.json")
        self.assertEqual({n: [e["sha256"], e["pool_digest"]] for n, e in f5["pools"].items() if e["eval_seed"] == 42},
                         V264_SEED42["f5"])
        self.assertEqual({n: M.dcfg.JOURNAL_D7_POOLS[n] for n in V264_SEED42["f5"]},
                         {n: v[0] for n, v in V264_SEED42["f5"].items()})
        f6 = load_json("tasks/harbor_mission/d7_pools/POOL_MANIFEST.json")
        self.assertEqual({e["file"]: [e["file_sha256"], e["pool_digest"]] for e in f6["pools"] if e["seed"] == 42},
                         V264_SEED42["f6"])
        for family, pins in V264_SEED42["appearing"].items():
            manifest, _ = M.apool.read_manifest(M.apool.family_pool_dir(family))
            for name, (sha, digest) in pins.items():
                self.assertEqual([manifest["pools"][name]["sha256"], manifest["pools"][name]["pool_digest"]],
                                 [sha, digest], name)
            self.assertEqual(sorted(manifest["provenance"]["kept_entries_from_previous_manifest"]), sorted(pins))

    def test_every_seed_task_vessel_is_pinned(self):
        count = 0
        for seed in SEEDS:
            for vessel in VESSELS:
                for task in HAZARD_TASKS:
                    name, pin, grid = f1_pin(seed, task, vessel)
                    self.assertIsNotNone(pin, (seed, task, vessel))
                    self.assertEqual(pin["file"], name)
                    self.assertEqual(grid["seed"], seed)
                    self.assertEqual(name.endswith("__seed%d.json.gz" % seed), seed != 42, name)
                table = f2_table(seed)
                self.assertEqual((table["eval_seed"], table["tier"]), (seed, 2))
                files = table["vessels"][vessel]["files"]
                self.assertEqual(sorted(files), ["0", "1", "2", "3", "4"])
                for dose, f in files.items():
                    self.assertEqual(f["file"].endswith("_D%s_seed%d.json.gz" % (dose, seed)), seed != 42, f["file"])
                name, pin = f3_pin(seed, vessel)
                self.assertEqual((pin["seed"], pin["vessel"]), (seed, vessel), name)
                path, sha = M.sk.journal_d7_pool_entry(vessel, 2, seed)
                self.assertIsNotNone(path, (seed, vessel))
                self.assertEqual(Path(path).parent.name, "d7_pools_approach")
                self.assertEqual(sha256_file(path), sha)                         # ships in the release
                self.assertEqual(M.js.StationD7Pool.load(path, sha).seed, seed)
                name, pin = f5_pin(seed, vessel)
                self.assertIsNotNone(pin, name)
                self.assertIsNotNone(f6_pin(seed, vessel), (seed, vessel))
                for task in M.aconf.PLACEMENT_TASKS:
                    family, name, entry = appearing_pin(task, vessel, seed)
                    self.assertIsNotNone(entry, name)
                    self.assertEqual((entry["task"], entry["vessel_id"], entry["seed"], entry["tier"]),
                                     (task, vessel, seed, 2))
                count += 1
        self.assertEqual(count, len(SEEDS) * len(VESSELS))
        self.assertEqual(len(D7_TASKS), 12)
        self.assertEqual(len(M.aconf.PLACEMENT_TASKS), 12)

    def test_a_seed_without_pins_resolves_to_nothing(self):
        for seed in (7, 11001, 41143):
            self.assertEqual(f1_pin(seed, "reach", "blueboat_v2")[1], None)
            self.assertEqual(f2_table(seed), {})
            self.assertEqual(f3_pin(seed, "blueboat_v2")[1], None)
            self.assertEqual(M.sk.journal_d7_pool_entry("blueboat_v2", 2, seed), (None, None))
            self.assertEqual(f5_pin(seed, "blueboat_v2")[1], None)
            self.assertIsNone(f6_pin(seed, "blueboat_v2"))
        self.assertEqual(M.jh.pool_file_name("reach", "blueboat_v2"), "reach__blueboat_v2.json.gz")
        self.assertEqual(M.jh.pool_file_name("reach", "blueboat_v2", 42), "reach__blueboat_v2.json.gz")
        self.assertEqual(M.jh.pool_file_name("reach", "blueboat_v2", 41041), "reach__blueboat_v2__seed41041.json.gz")

    def test_station_keeping_env_uses_the_entry_of_cfg_seed(self):
        for vessel in VESSELS:
            legacy = M.sk.journal_d7_pool_entry(vessel, 2)                      # what __post_init__ stores
            self.assertEqual(legacy, M.sk.journal_d7_pool_entry(vessel, 2, 42))
            for seed in SEEDS:
                cfg = SimpleNamespace(vehicle=vessel, seed=seed, journal_d7_pool=legacy[0],
                                      journal_d7_pool_sha256=legacy[1])
                self.assertEqual(M.sk.journal_d7_pool_for_seed(cfg, 2), M.sk.journal_d7_pool_entry(vessel, 2, seed))
            for seed in (7, 11001):                                              # unpinned: unchanged (check_env refuses)
                cfg = SimpleNamespace(vehicle=vessel, seed=seed, journal_d7_pool=legacy[0],
                                      journal_d7_pool_sha256=legacy[1])
                self.assertEqual(M.sk.journal_d7_pool_for_seed(cfg, 2), legacy)
            explicit = SimpleNamespace(vehicle=vessel, seed=41041, journal_d7_pool="elsewhere/pool.json",
                                       journal_d7_pool_sha256="0" * 64)
            self.assertEqual(M.sk.journal_d7_pool_for_seed(explicit, 2), ("elsewhere/pool.json", "0" * 64))
            pool = object()
            held = SimpleNamespace(vehicle=vessel, seed=41041, journal_d7_pool=pool, journal_d7_pool_sha256=None)
            self.assertIs(M.sk.journal_d7_pool_for_seed(held, 2)[0], pool)

    def test_docking_blueboat_d1_not_applicable(self):
        self.assertEqual(M.ladder.not_applicable_doses("docking", "blueboat_v2"), (1,))
        self.assertEqual(M.ladder.not_applicable_doses("docking", "wamv_vrx_v2_opt1"), ())
        for seed in FORMAL_SEEDS:
            _, name, entry = appearing_pin("docking", "blueboat_v2", seed)
            self.assertEqual(entry["per_dose_counts"][1], [0, 0, 0], name)


@unittest.skipUnless(os.environ.get("USVBENCH_D7_POOL_DIR"), "set USVBENCH_D7_POOL_DIR=<G49 bundle> for the resolve tests")
class Resolve(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle = Path(os.environ["USVBENCH_D7_POOL_DIR"]).resolve()
        os.environ.pop("USVBENCH_HARBOUR_D7_POOL", None)

    def inside(self, path):
        self.assertEqual(Path(path).resolve().parent, self.bundle, path)

    def test_bundle_is_the_g49_bundle(self):
        self.assertEqual(sha256_file(self.bundle / "BUNDLE_MANIFEST.txt"), G49_BUNDLE_MANIFEST_SHA256)

    def test_f1_hazard_nav(self):
        for seed in SEEDS:
            for vessel in VESSELS:
                for task in HAZARD_TASKS:
                    with self.subTest(seed=seed, task=task, vessel=vessel):
                        name, pin, _ = f1_pin(seed, task, vessel)
                        cfg = SimpleNamespace(journal_task=task, vehicle=vessel, journal_geometry_tier=2, seed=seed,
                                              journal_d7_pool=None, journal_d7_pool_dir=str(self.bundle))
                        pool, meta = M.jh.resolve_d7_pool(cfg)
                        self.inside(self.bundle / name)
                        self.assertEqual((meta["file_sha256"], pool.pool_digest), (pin["sha256"], pin["pool_digest"]))
                        self.assertEqual(meta["seed"], seed)
                        if seed == 42:
                            self.assertEqual([meta["file_sha256"], pool.pool_digest], V264_SEED42["f1"][name])

    def test_f2_path_hazard(self):
        for seed in SEEDS:
            for vessel in VESSELS:
                entry = f2_table(seed)["vessels"][vessel]
                for dose in range(5):
                    with self.subTest(seed=seed, vessel=vessel, dose=dose):
                        if dose == 0:
                            suite = M.dynamic.DynamicSuiteConfig(vessel_id=vessel)
                        else:
                            suite = M.jconf.resolve_journal_conditions(presets=["dynamic=D%d" % dose],
                                                                       seed=seed).dynamic_config(vessel)
                        pool = M.ph.resolve_pool("", vessel=vessel, tier=2, suite=suite,
                                                 path_variable=str(self.bundle), seed=seed)
                        self.inside(self.bundle / entry["files"][str(dose)]["file"])
                        self.assertEqual((pool.eval_seed, pool.pool_digest), (seed, entry["pool_digest"]))
                        if seed == 42:
                            self.assertEqual(pool.pool_digest, V264_SEED42["f2"][vessel]["pool_digest"])
                            self.assertEqual(entry["files"][str(dose)]["sha256"],
                                             V264_SEED42["f2"][vessel]["files"][str(dose)][1])
                        self.assertEqual(sha256_file(self.bundle / entry["files"][str(dose)]["file"]),
                                         entry["files"][str(dose)]["sha256"])

    def test_f3_path_following(self):
        for seed in SEEDS:
            for vessel in VESSELS:
                with self.subTest(seed=seed, vessel=vessel):
                    name, pin = f3_pin(seed, vessel)
                    cfg = SimpleNamespace(journal_d7_pool_path=None, vehicle=vessel, seed=seed,
                                          journal_geometry_tier=2, journal_d7_pool_sha256=None)
                    pool, source = M.pf.load_journal_d7_pool(cfg)
                    path, _ = M.pf.resolve_pool_path(cfg)
                    self.inside(path)
                    self.assertEqual(source, "env")
                    self.assertEqual(pool.pool_digest, pin["pool_digest"])
                    self.assertEqual(sha256_file(path), pin["file_sha256"])
                    if seed == 42:
                        self.assertEqual([pin["file_sha256"], pin["container_sha256"], pool.pool_digest],
                                         V264_SEED42["f3"][name])

    def test_f4_station_keeping(self):
        index = load_json("tasks/station_keeping/d7_pools_approach/INDEX.json")["pools"]
        for seed in SEEDS:
            for vessel in VESSELS:
                with self.subTest(seed=seed, vessel=vessel):
                    path, sha = M.sk.journal_d7_pool_entry(vessel, 2, seed)
                    pool = M.js.StationD7Pool.load(path, sha)
                    pool.check_env(vessel=vessel, seed=seed, tier=2, num_envs=4)
                    key = "%s/tier2" % vessel if seed == 42 else "%s/tier2/seed%d" % (vessel, seed)
                    self.assertEqual(pool.pool_digest, index[key]["pool_digest"])
                    self.assertTrue(M.sk.journal_pinned_pool_fields(pool, vessel, 2, seed))   # D19 stamp present
                    self.assertEqual(sha256_file(self.bundle / index[key]["file"]), sha)          # bundle copy
                    self.assertEqual(sha256_file(self.bundle / index[key]["witness_file"]), index[key]["witness_sha256"])
                    if seed == 42:
                        self.assertEqual([index[key]["file"], sha, index[key]["witness_sha256"], pool.pool_digest],
                                         V264_SEED42["f4"][key])
                        self.assertTrue(M.sk.journal_pinned_pool_fields(pool, vessel, 2))   # v2.6.4 call form

    def test_f5_docking(self):
        manifest = load_json("tasks/docking/journal_pools/POOL_MANIFEST.json")["pools"]
        for seed in SEEDS:
            for vessel in VESSELS:
                with self.subTest(seed=seed, vessel=vessel):
                    name, pin = f5_pin(seed, vessel)
                    pool = M.dpool.load_pool(self.bundle / name, expected_sha256=pin, vessel=vessel, tier=2,
                                             seed=seed, berth_walls=True)
                    self.assertEqual((pool.eval_seed, pool.pool_digest), (seed, manifest[name]["pool_digest"]))
                    if seed == 42:
                        self.assertEqual([pin, pool.pool_digest], V264_SEED42["f5"][name])

    def test_f6_harbour(self):
        for seed in SEEDS:
            for vessel in VESSELS:
                with self.subTest(seed=seed, vessel=vessel):
                    cfg = SimpleNamespace(vehicle=vessel, seed=seed, journal_d7_pool_path=None)
                    path, source = M.hpool.resolve_pool_source(cfg)
                    self.inside(path)
                    self.assertEqual(source, "USVBENCH_D7_POOL_DIR")
                    raw, meta = M.hpool.load_pool(path, M.dynamic, vessel=vessel)
                    pin = f6_pin(seed, vessel)
                    self.assertTrue(meta["pinned"])
                    self.assertEqual((int(meta["seed"]), int(meta["tier"])), (seed, 2))
                    self.assertEqual((meta["file_sha256"], raw.pool_digest), (pin["file_sha256"], pin["pool_digest"]))
                    if seed == 42:
                        self.assertEqual([meta["file_sha256"], raw.pool_digest], V264_SEED42["f6"][Path(path).name])

    def test_appearing(self):
        for seed in SEEDS:
            for vessel in VESSELS:
                for task in M.aconf.PLACEMENT_TASKS:
                    with self.subTest(seed=seed, task=task, vessel=vessel):
                        family, name, entry = appearing_pin(task, vessel, seed)
                        self.assertEqual(sha256_file(self.bundle / name), entry["sha256"], name)
                        if seed == 42:
                            # the seed-42 appearing pins are the v2.6.4 ones (admitted under the v6m ladder, before
                            # the frozen v7m doses): same file and digest as in v2.6.4, still refused for a v7m run
                            self.assertEqual([entry["sha256"], entry["pool_digest"]],
                                             V264_SEED42["appearing"][family][name])
                            if (task, vessel) == ("docking", "blueboat_v2"):
                                # it offers D1, which the G48 patch made not applicable by design: refused on load
                                with self.assertRaisesRegex(ValueError, "required dose"):
                                    M.apool.load_pinned_pool(family, task, vessel, seed, 2, runtime_dir=self.bundle,
                                                             runtime_source="test")
                                continue
                        pool, record = M.apool.load_pinned_pool(family, task, vessel, seed, 2,
                                                                runtime_dir=self.bundle, runtime_source="test")
                        self.inside(record["path"])
                        self.assertEqual((record["sha256"], pool.pool_digest), (entry["sha256"], entry["pool_digest"]))
                        if seed == 42:
                            self.assertNotEqual(pool.suite_identity["version"], M.aconf.APPEARING_SUITE_VERSION)
                        else:
                            self.assertEqual(pool.suite_identity["version"], M.aconf.APPEARING_SUITE_VERSION)


if __name__ == "__main__":
    unittest.main(verbosity=2)
