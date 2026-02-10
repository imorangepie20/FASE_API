#!/bin/bash
# Kuka Spotify 추천 시스템 테스트 명령어
# 서버: http://localhost:8000

echo "=== 1. 모델 정보 확인 ==="
curl -s "http://localhost:8000/api/spotify/models" | python3 -m json.tool

echo ""
echo "=== 2. BTS 테스트 (Ensemble, 기본) ==="
curl -s "http://localhost:8000/api/spotify/recommend?artist=BTS&k=5" | python3 -m json.tool

echo ""
echo "=== 3. BTS 테스트 (KNN, 오디오만) ==="
curl -s "http://localhost:8000/api/spotify/recommend?artist=BTS&k=5&model=knn" | python3 -m json.tool

echo ""
echo "=== 4. Miles Davis 테스트 (Jazz) ==="
curl -s "http://localhost:8000/api/spotify/recommend?artist=Miles%20Davis&k=5" | python3 -m json.tool

echo ""
echo "=== 5. Nirvana 테스트 (Grunge) ==="
curl -s "http://localhost:8000/api/spotify/recommend?artist=Nirvana&k=5" | python3 -m json.tool

echo ""
echo "=== 6. Taylor Swift 테스트 (Pop) ==="
curl -s "http://localhost:8000/api/spotify/recommend?artist=Taylor%20Swift&k=5" | python3 -m json.tool

echo ""
echo "=== 7. 텍스트 모델 테스트 ==="
curl -s "http://localhost:8000/api/spotify/recommend?artist=BTS&k=5&model=text" | python3 -m json.tool

echo ""
echo "=== 8. 하이브리드 모델 테스트 ==="
curl -s "http://localhost:8000/api/spotify/recommend?artist=BTS&k=5&model=hybrid" | python3 -m json.tool

echo ""
echo "=== 테스트 완료 ==="
