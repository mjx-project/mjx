set -eu

TEST_DIR=$(pwd)
TENHOU_DIR=${TEST_DIR}/../tenhou

echo "################################################"
echo "# 天鳳位"
echo "################################################"

mkdir ${TEST_DIR}/tmp

# Download
echo "* Downloading ..."
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
echo "* Unzipping ..."
cd ${TEST_DIR}/tmp
for zip_file in $(ls); do unzip ${zip_file} &>/dev/null && rm -rf ${zip_file}; done
for dir in $(ls); do
  cd ${dir}
  for x in $(ls); do mv ${x} ${x}.gz; done
  cd ../
  gzip -dr ${dir} &>/dev/null
done

# Apply filter  TODO: replace here by python script
echo "* Filtering ..."
cd ${TEST_DIR}/tmp
for dir in $(ls); do
  cd ${dir}
  for file in $(ls); do
    if [[ $(ls | wc -l) -lt 100 ]]; then break; fi
    rm ${file}
  done
  cd ../
done

# Move to mjlog from tmp dir
echo "* Moving ..."
cd ${TEST_DIR}/tmp
for dir in $(ls); do
  mv ${dir}/*.mjlog ${TEST_DIR}/resources/mjlog/
done

# Converting mjlog 
echo "* Converting ..."
cd ${TEST_DIR}/tmp
for dir in $(ls); do
  cd ${dir}
  python3 ${TENHOU_DIR}/mjlog_decoder.py ${TEST_DIR}/resources/mjlog ${TEST_DIR}/resources/json --modify
  cd ../
done

cd ${TEST_DIR}
rm -rf ${TEST_DIR}/tmp

