set -eu

TEST_DIR=$(pwd)
TENHOU_DIR=${TEST_DIR}/../tenhou

mkdir -p ${TEST_DIR}/resources/zip
mkdir -p ${TEST_DIR}/tmp
# trap finally EXIT

function download {
	ix=$1
	if [[ ! -e mjlog_pf4-20_n${ix}.zip ]]; 
		then curl -O https://tenhou.net/0/log/mjlog_pf4-20_n${ix}.zip; 
	fi
}

function check_gz {
  echo ".gz: $(ls ${TEST_DIR}/tmp | grep ".gz$" | wc -l), .mjlog: $(ls ${TEST_DIR}/tmp | grep ".mjlog$" | wc -l)"
}

function finally {
	rm -rf ${TEST_DIR}/tmp
}

# 天鳳位のデータのみ TODO: 他の鳳凰卓のデータ
echo "* Downloading ..."
cd ${TEST_DIR}/resources/zip
for ix in $(seq 1 17); do download ${ix}; done

echo "* Unzipping ..."
check_gz
cd ${TEST_DIR}/resources/zip
for zip_file in $(ls); do unzip -j ${zip_file} -d ${TEST_DIR}/tmp &>/dev/null; done
check_gz
cd ${TEST_DIR}/tmp
for x in $(ls); do mv ${x} ${x}.gz; done
check_gz
cd ${TEST_DIR}/tmp
for gzip_file in $(ls); do gzip -d ${gzip_file} &>/dev/null || true ; done
check_gz
 
echo "* Filtering ..."
python3 ${TENHOU_DIR}/filter.py ${TEST_DIR}/tmp --hounan --rm-users "o(>ロ<*)o"
  
echo "* Converting ..."
python3 ${TENHOU_DIR}/mjlog_decoder.py ${TEST_DIR}/tmp ${TEST_DIR}/resources/json --modify
 
