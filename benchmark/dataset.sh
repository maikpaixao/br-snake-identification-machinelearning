mkdir dataset
#wget -P dataset https://thunlp.oss-cn-qingdao.aliyuncs.com/opennre/pretrain/bert-base-uncased/config.json
wget -P dataset --load-cookies /tmp/cookies.txt --no-check-certificate "https://docs.google.com/uc?export=download&confirm=$(wget 
--quiet --save-cookies /tmp/cookies.txt --keep-session-cookies 
--no-check-certificate 'https://docs.google.com/uc?export=download&id=1u4t8zdXQybSILgk_pbTYELc3S7HMPJ3l' 
-O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+). */\1\n/p')&id=1u4t8zdXQybSILgk_pbTYELc3S7HMPJ3l" 
-O dataset.zip && rm -rf /tmp/cookies.txt