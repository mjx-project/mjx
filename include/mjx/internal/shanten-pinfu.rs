use std::collections::{VecDeque, BTreeSet};
use std::fs;
use std::io::Write;
use std::time::Instant;

fn encode(hand: &Vec<usize>, draw: usize) -> usize {
    let mut code: usize = 0;
    for h in hand.iter() {
        code = 5 * code + h;
    }
    code * 10 + draw
}

// x面子y雀頭z両面完成の形を列挙
fn complete_hands(x: usize, y: usize, z: usize) -> BTreeSet<(Vec<usize>,usize)> {
    assert!(x <= 4);
    assert!(y <= 1);
    assert!(z <= 1);

    // 面子のリスト
    let sets = vec![
        vec![1, 1, 1, 0, 0, 0, 0, 0, 0],
        vec![0, 1, 1, 1, 0, 0, 0, 0, 0],
        vec![0, 0, 1, 1, 1, 0, 0, 0, 0],
        vec![0, 0, 0, 1, 1, 1, 0, 0, 0],
        vec![0, 0, 0, 0, 1, 1, 1, 0, 0],
        vec![0, 0, 0, 0, 0, 1, 1, 1, 0],
        vec![0, 0, 0, 0, 0, 0, 1, 1, 1],
    ];

    // 両面待ちのリスト
    let open_ends = if z == 1 {
        vec![
            (vec![0,1,1,0,0,0,0,0,0], 0),
            (vec![0,1,1,0,0,0,0,0,0], 3),
            (vec![0,0,1,1,0,0,0,0,0], 1),
            (vec![0,0,1,1,0,0,0,0,0], 4),
            (vec![0,0,0,1,1,0,0,0,0], 2),
            (vec![0,0,0,1,1,0,0,0,0], 5),
            (vec![0,0,0,0,1,1,0,0,0], 3),
            (vec![0,0,0,0,1,1,0,0,0], 6),
            (vec![0,0,0,0,0,1,1,0,0], 4),
            (vec![0,0,0,0,0,1,1,0,0], 7),
            (vec![0,0,0,0,0,0,1,1,0], 5),
            (vec![0,0,0,0,0,0,1,1,0], 8),
        ]
    } else {
        vec![
            (vec![0,0,0,0,0,0,0,0,0], 9),
        ]
    };

    // 雀頭のリスト
    let heads = if y == 1 {
        vec![
            vec![2, 0, 0, 0, 0, 0, 0, 0, 0],
            vec![0, 2, 0, 0, 0, 0, 0, 0, 0],
            vec![0, 0, 2, 0, 0, 0, 0, 0, 0],
            vec![0, 0, 0, 2, 0, 0, 0, 0, 0],
            vec![0, 0, 0, 0, 2, 0, 0, 0, 0],
            vec![0, 0, 0, 0, 0, 2, 0, 0, 0],
            vec![0, 0, 0, 0, 0, 0, 2, 0, 0],
            vec![0, 0, 0, 0, 0, 0, 0, 2, 0],
            vec![0, 0, 0, 0, 0, 0, 0, 0, 2],
        ]
    } else {
        vec![
            vec![0, 0, 0, 0, 0, 0, 0, 0, 0],
        ]
    };

    let mut ret: BTreeSet<(Vec<usize>,usize)> = BTreeSet::new();

    let mut sid = vec![0; x+1];

    while sid[x] == 0 {
        for head in heads.iter() {
            for open_end in open_ends.iter() {
                let hand: Vec<usize> = (0..9).map(|i| {
                    (0..x).map(|j| sets[sid[j]][i]).sum::<usize>() + head[i] + open_end.0[i]
                }).collect();
                let draw: usize = open_end.1;

                if (0..9).all(|i| hand[i] + if draw == i { 1 } else { 0 } <= 4) {
                    ret.insert((hand, draw));
                }
            }
        }

        let mut i = 0;
        sid[i] += 1;
        while sid[i] == sets.len() {
            sid[i] = 0;
            i += 1;
            sid[i] += 1;
        }
    }

    ret
}

// 01BFSを用いてシャンテン数を計算
fn bfs(ws: BTreeSet<(Vec<usize>,usize)>) -> Vec<i32> {
    let mut dist = vec![i32::MAX; 19531250];
    let mut deq: VecDeque<(i32,Vec<usize>,usize)> = VecDeque::new();

    for (hand, draw) in ws {
        dist[encode(&hand, draw)] = 0;
        deq.push_back((0, hand, draw));
    }

    while !deq.is_empty() {
        let (d, hand, draw) = deq.pop_front().unwrap();
        if d > dist[encode(&hand, draw)] {
            continue;
        }

        // (hand, k) -> (hand + k, None)
        if draw == 9 {
            for k in 0..9 {
                if hand[k] > 0 {
                    let mut hand_tmp = hand.clone();
                    hand_tmp[k] -= 1;
                    let code = encode(&hand_tmp, k);
                    if dist[code] > d {
                        dist[code] = d;
                        deq.push_front((d, hand_tmp, k));
                    }
                }
            }
        }

        // (hand, None) -> (hand - k, None)
        if draw == 9 {
            for k in 0..9 {
                if hand[k] < 4 {
                    let mut hand_tmp = hand.clone();
                    hand_tmp[k] += 1;
                    let code = encode(&hand_tmp, 9); 
                    if dist[code] > d {
                        dist[code] = d;
                        deq.push_front((d, hand_tmp, 9));
                    }
                }
            }
        }

        // (hand, None) -> (hand, draw)
        if draw != 9 {
            let code = encode(&hand, 0);
            if dist[code] > d + 1 {
                dist[code] = d + 1;
                deq.push_back((d + 1, hand.clone(), 9));
            }
        }
    }

    dist
}

fn main() {
    let start = Instant::now();
    let shanten: Vec<Vec<Vec<Vec<i32>>>> = (0..4).map(|x| {
        (0..2).map(|y| {
            (0..2).map(|z| {
                let ws = complete_hands(x, y, z);
                bfs(ws)
            }).collect()
        }).collect()
    }).collect();
    dbg!(shanten[0][0][0].len());
    // assert!(shanten[0][0][0].iter().all(|d| *d == 0));
    for code in 0..1953125 {
        assert!(shanten[0][0][0][code * 10 + 9] == 0);
    }

    // 計算結果を保存
    let mut f = fs::File::create("shanten-pinfu.txt").unwrap();
    for code in 0..19531250 {
        f.write_all(format!("{} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {}\n",
            shanten[0][0][0][code],
            shanten[1][0][0][code],
            shanten[2][0][0][code],
            shanten[3][0][0][code],
            shanten[0][1][0][code],
            shanten[1][1][0][code],
            shanten[2][1][0][code],
            shanten[3][1][0][code],
            shanten[0][0][1][code],
            shanten[1][0][1][code],
            shanten[2][0][1][code],
            shanten[3][0][1][code],
            shanten[0][1][1][code],
            shanten[1][1][1][code],
            shanten[2][1][1][code],
            shanten[3][1][1][code],
        ).as_bytes()).unwrap();
    }
    let elapsed_time = start.elapsed().as_nanos() as f64 
        / 1_000_000_000 as f64;
    println!("elapsed_time: {} [sec]", elapsed_time);
}
