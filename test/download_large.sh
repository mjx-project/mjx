set -eu

TEST_DIR="$(pwd)"
TENHOU_DIR=${TEST_DIR}/../tenhou
ZIP_DIR=${TEST_DIR}/resources/zip
TMP_DIR=${TEST_DIR}/resources/tmp

mkdir -p ${ZIP_DIR}
mkdir -p ${TMP_DIR}
trap finally EXIT

function download {
	ix=$1
	if [[ ! -e mjlog_pf4-20_n${ix}.zip ]]; 
		then curl -O https://tenhou.net/0/log/mjlog_pf4-20_n${ix}.zip; 
	fi
}

function check_gz {
  echo ".gz: $(ls ${TMP_DIR} | grep ".gz$" | wc -l), .mjlog: $(ls ${TMP_DIR} | grep ".mjlog$" | wc -l)"
}

function finally {
	rm -rf ${TMP_DIR}
}

# 天鳳位のデータのみ TODO: 他の鳳凰卓のデータ
echo "* Downloading ..."
cd ${ZIP_DIR}
for ix in $(seq 1 17); do download ${ix}; done

echo "* Unzipping ..."
check_gz
cd ${ZIP_DIR}
for zip_file in $(ls); do unzip -j ${zip_file} -d ${TMP_DIR} &>/dev/null; done
check_gz
cd ${TMP_DIR}
for x in $(ls); do mv ${x} ${x}.gz; done
check_gz
cd ${TMP_DIR}
for gzip_file in $(ls); do gzip -d ${gzip_file} &>/dev/null || true ; done
check_gz
 
echo "* Filtering ..."
python3 ${TENHOU_DIR}/filter.py ${TMP_DIR} --hounan --rm-users "o(>ロ<*)o" "<>" ">_0@杏杏軍團" "HKG<>LHR"
  
echo "* Converting ..."
python3 ${TENHOU_DIR}/mjlog_decoder.py ${TMP_DIR} ${TEST_DIR}/resources/json --modify
 
