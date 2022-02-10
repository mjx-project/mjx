use std::collections::{VecDeque, BTreeSet};
use std::fs;
use std::io::Write;
use std::time::Instant;

fn encode(hand: &Vec<usize>) -> usize {
    let mut code: usize = 0;
    for h in hand.iter() {
        code = 5 * code + h;
    }
    code
}

// x面子y雀頭z両面の形を列挙
fn complete_hands(x: usize, y: usize, z: usize) -> BTreeSet<Vec<usize>> {
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
            vec![0,1,1,0,0,0,0,0,0],
            vec![0,1,1,0,0,0,0,0,0],
            vec![0,0,1,1,0,0,0,0,0],
            vec![0,0,1,1,0,0,0,0,0],
            vec![0,0,0,1,1,0,0,0,0],
            vec![0,0,0,1,1,0,0,0,0],
            vec![0,0,0,0,1,1,0,0,0],
            vec![0,0,0,0,1,1,0,0,0],
            vec![0,0,0,0,0,1,1,0,0],
            vec![0,0,0,0,0,1,1,0,0],
            vec![0,0,0,0,0,0,1,1,0],
            vec![0,0,0,0,0,0,1,1,0],
        ]
    } else {
        vec![
            vec![0,0,0,0,0,0,0,0,0],
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

    let mut ret: BTreeSet<Vec<usize>> = BTreeSet::new();

    let mut sid = vec![0; x+1];

    while sid[x] == 0 {
        for head in heads.iter() {
            for open_end in open_ends.iter() {
                let hand: Vec<usize> = (0..9).map(|i| {
                    (0..x).map(|j| sets[sid[j]][i]).sum::<usize>() + head[i] + open_end[i]
                }).collect();

                if (0..9).all(|i| hand[i] <= 4) {
                    ret.insert(hand);
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
fn bfs(ws: BTreeSet<Vec<usize>>) -> Vec<i32> {
    let mut dist = vec![i32::MAX; 1953125];
    let mut deq: VecDeque<(i32,Vec<usize>)> = VecDeque::new();

    for hand in ws {
        dist[encode(&hand)] = 0;
        deq.push_back((0, hand));
    }

    while !deq.is_empty() {
        let (d, hand) = deq.pop_front().unwrap();
        if d > dist[encode(&hand)] {
            continue;
        }

        for k in 0..9 {
            if hand[k] < 4 {
                let mut hand_add = hand.clone();
                hand_add[k] += 1;
                let code_add = encode(&hand_add);
                if dist[code_add] > d {
                    dist[code_add] = d;
                    deq.push_front((d, hand_add));
                }
            }
            if hand[k] > 0 {
                let mut hand_sub = hand.clone();
                hand_sub[k] -= 1;
                let code_sub = encode(&hand_sub);
                if dist[code_sub] > d + 1 {
                    dist[code_sub] = d + 1;
                    deq.push_back((d + 1, hand_sub));
                }
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
    assert!(shanten[0][0][0].iter().all(|d| *d == 0));

    // 計算結果を保存
    let mut f = fs::File::create("shanten-pinfu.txt").unwrap();
    for code in 0..1953125 {
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
