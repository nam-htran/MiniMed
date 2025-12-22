#!/bin/bash

VOLUME_NAME="medcot_primekg_data"

echo "🛑 Stopping Neo4j container if running..."
docker-compose down

echo "🗑️  Deleting old Neo4j data volume: $VOLUME_NAME..."
docker volume rm $VOLUME_NAME || true

echo "🚀 Starting PrimeKG data import into Neo4j (WITH GDS PLUGIN on 5.26.18)..."

MSYS_NO_PATHCONV=1 docker run --interactive --tty --rm \
    --volume "$(pwd)/data/primekg/import":/import \
    --volume $VOLUME_NAME:/data \
    --env NEO4J_PLUGINS='["apoc", "graph-data-science"]' \
    neo4j:5.26.18 \
    neo4j-admin database import full \
    --nodes=/import/nodes.csv \
    --relationships=/import/edges.csv \
    --overwrite-destination \
    neo4j
# -----------------------------------------------------------------------------

if [ $? -eq 0 ]; then
    echo "✅ IMPORT DỮ LIỆU THÀNH CÔNG!"
    echo "👍 Dữ liệu đã được nạp vào volume '$VOLUME_NAME'."
    echo "🚀 Tự động khởi động Neo4j server bằng docker-compose..."
    
    docker-compose up -d

    echo "⏳ Đang đợi server khởi động (khoảng 15-20 giây)..."
    sleep 20

    echo "✅✅✅ HOÀN TẤT! Server Neo4j đã được khởi động và sẵn sàng."
    echo "👉 Bây giờ bạn có thể chạy 'python main.py --query \"...\"'"

else
    echo "❌ IMPORT DỮ LIỆU THẤT BẠI. Vui lòng kiểm tra lỗi ở trên."
    exit 1
fi