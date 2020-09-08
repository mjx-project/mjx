set -eu

TEST_DIR=$(pwd)

################################################
# 天鳳位
################################################

mkdir ${TEST_DIR}/tmp

# Download
cd ${TEST_DIR}/tmp
curl -O https://tenhou.net/0/log/mjlog_pf4-20_n1.zip  # CLS
# curl -O https://tenhou.net/0/log/mjlog_pf4-20_n16.zip # 藤井聡ふと
# curl -O https://tenhou.net/0/log/mjlog_pf4-20_n15.zip # 右折するひつじ
# curl -O https://tenhou.net/0/log/mjlog_pf4-20_n14.zip # お知らせ
# curl -O https://tenhou.net/0/log/mjlog_pf4-20_n13.zip # gousi
# curl -O https://tenhou.net/0/log/mjlog_pf4-20_n12.zip # おかもと
# curl -O https://tenhou.net/0/log/mjlog_pf4-20_n11.zip # トトリ先生19歳
# curl -O https://tenhou.net/0/log/mjlog_pf4-20_n10.zip # ウルトラ立直
# curl -O https://tenhou.net/0/log/mjlog_pf4-20_n9.zip  # 就活生@川村軍団
# curl -O https://tenhou.net/0/log/mjlog_pf4-20_n8.zip  # かにマジン
# curl -O https://tenhou.net/0/log/mjlog_pf4-20_n7.zip  # コーラ下さい
# curl -O https://tenhou.net/0/log/mjlog_pf4-20_n6.zip  # タケオしゃん
# curl -O https://tenhou.net/0/log/mjlog_pf4-20_n5.zip  # 太くないお
# curl -O https://tenhou.net/0/log/mjlog_pf4-20_n4.zip  # すずめクレイジー
# curl -O https://tenhou.net/0/log/mjlog_pf4-20_n3.zip  # 独歩
# curl -O https://tenhou.net/0/log/mjlog_pf4-20_n2.zip  # （≧▽≦）
# curl -O https://tenhou.net/0/log/mjlog_pf4-20_n1.zip  # ASAPIN

# Unzip 
cd ${TEST_DIR}/tmp
for zip_file in $(ls); do unzip ${zip_file} && rm -rf ${zip_file} done
for dir in $(ls); do
  cd ${dir}
  for x in $(ls); do mv ${x} ${x}.gz; done
  cd ../
  gzip -dr ${dir}
done

# TODO: Apply filter

cd ${TEST_DIR}
rm -rf ${TEST_DIR}/tmp
