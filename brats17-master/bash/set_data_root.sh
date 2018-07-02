#set_data_root.sh CONFIG_DIR DATA_ROOT_PATH
sed -ri "s|(data_root *= *).*|\1$2|" $(grep -lr data_root $1)
