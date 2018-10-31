# wget https://www.w3.org/TR/PNG/iso_8859-1.txt
# mv iso_8859-1.txt input.txt



# Kaggle

echo ""
echo "+--------------------------------------------------------------------"
echo "| World Cup Datasets from Kaggle"
echo "+--------------------------------------------------------------------"
echo "| Author: Jiho Choi (jihochoi@snu.ac.kr)"
echo "|"
echo "| Kaggle requires its own CLI and authentication"
echo "|   for the simplicity, I backed up the datasets in my personal repositories"
echo "|"
echo "| Origin of the datasets"
echo "|   - https://www.kaggle.com/tadhgfitzgerald/fifa-international-soccer-mens-ranking-1993now"
echo "|   - https://www.kaggle.com/djamshed/fifa-world-cup-2018-players"
echo "|"
echo "+--------------------------------------------------------------------"
echo ""

wget https://jihochoi.github.io/datasets/fifa_ranking.csv
wget https://jihochoi.github.io/datasets/wc2018_players.csv

rm -r ./data
mkdir ./data

echo ""
echo "Locate Files"
mv ./fifa_ranking.csv ./data/fifa_ranking.csv
mv ./wc2018_players.csv ./data/wc2018_players.csv
ls ./data

echo ""
echo "Remove Headers"
tail -n +2 ./data/fifa_ranking.csv > ./data/ranking.csv
tail -n +2 ./data/wc2018_players.csv > ./data/players.csv
ls ./data
