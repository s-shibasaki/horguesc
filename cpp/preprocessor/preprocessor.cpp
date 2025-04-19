#include <iostream>
#include <fstream>
#include <limits>
#include <iomanip> // std::setprecision のために追加
#include "jvdata.h"
#include <algorithm>

using namespace std::literals;

int main(int argc, char *argv[])
{
	if (argc == 1)
	{
		std::cerr << "Usage: " << argv[0] << " [hello|testinteger|testfloat|testdata]" << std::endl;
		return 1;
	}

	std::string_view arg1(argv[1]);

	if (arg1 == "hello")
	{
		std::cout << "Hello, World!" << std::endl;
		return 0;
	}

	else if (arg1 == "testinteger")
	{
		std::cout << "整数型の情報：" << std::endl;
		std::cout << "========================================" << std::endl;

		// 標準整数型の範囲
		std::cout << "【標準整数型の範囲】" << std::endl;
		std::cout << "----------------------------------------" << std::endl;
		std::cout << "char:           " << std::setw(20) << static_cast<int>(std::numeric_limits<char>::min())
				  << " から " << std::setw(20) << static_cast<int>(std::numeric_limits<char>::max()) << std::endl;
		std::cout << "signed char:    " << std::setw(20) << static_cast<int>(std::numeric_limits<signed char>::min())
				  << " から " << std::setw(20) << static_cast<int>(std::numeric_limits<signed char>::max()) << std::endl;
		std::cout << "unsigned char:  " << std::setw(20) << static_cast<int>(std::numeric_limits<unsigned char>::min())
				  << " から " << std::setw(20) << static_cast<int>(std::numeric_limits<unsigned char>::max()) << std::endl;
		std::cout << "short:          " << std::setw(20) << std::numeric_limits<short>::min()
				  << " から " << std::setw(20) << std::numeric_limits<short>::max() << std::endl;
		std::cout << "unsigned short: " << std::setw(20) << std::numeric_limits<unsigned short>::min()
				  << " から " << std::setw(20) << std::numeric_limits<unsigned short>::max() << std::endl;
		std::cout << "int:            " << std::setw(20) << std::numeric_limits<int>::min()
				  << " から " << std::setw(20) << std::numeric_limits<int>::max() << std::endl;
		std::cout << "unsigned int:   " << std::setw(20) << std::numeric_limits<unsigned int>::min()
				  << " から " << std::setw(20) << std::numeric_limits<unsigned int>::max() << std::endl;
		std::cout << "long:           " << std::setw(20) << std::numeric_limits<long>::min()
				  << " から " << std::setw(20) << std::numeric_limits<long>::max() << std::endl;
		std::cout << "unsigned long:  " << std::setw(20) << std::numeric_limits<unsigned long>::min()
				  << " から " << std::setw(20) << std::numeric_limits<unsigned long>::max() << std::endl;
		std::cout << "long long:      " << std::setw(20) << std::numeric_limits<long long>::min()
				  << " から " << std::setw(20) << std::numeric_limits<long long>::max() << std::endl;
		std::cout << "unsigned long long: " << std::setw(16) << std::numeric_limits<unsigned long long>::min()
				  << " から " << std::setw(20) << std::numeric_limits<unsigned long long>::max() << std::endl;

		// 固定幅整数型の範囲
		std::cout << "\n【固定幅整数型の範囲】" << std::endl;
		std::cout << "----------------------------------------" << std::endl;
		std::cout << "int8_t:         " << std::setw(20) << static_cast<int>(std::numeric_limits<int8_t>::min())
				  << " から " << std::setw(20) << static_cast<int>(std::numeric_limits<int8_t>::max()) << std::endl;
		std::cout << "uint8_t:        " << std::setw(20) << static_cast<int>(std::numeric_limits<uint8_t>::min())
				  << " から " << std::setw(20) << static_cast<int>(std::numeric_limits<uint8_t>::max()) << std::endl;
		std::cout << "int16_t:        " << std::setw(20) << std::numeric_limits<int16_t>::min()
				  << " から " << std::setw(20) << std::numeric_limits<int16_t>::max() << std::endl;
		std::cout << "uint16_t:       " << std::setw(20) << std::numeric_limits<uint16_t>::min()
				  << " から " << std::setw(20) << std::numeric_limits<uint16_t>::max() << std::endl;
		std::cout << "int32_t:        " << std::setw(20) << std::numeric_limits<int32_t>::min()
				  << " から " << std::setw(20) << std::numeric_limits<int32_t>::max() << std::endl;
		std::cout << "uint32_t:       " << std::setw(20) << std::numeric_limits<uint32_t>::min()
				  << " から " << std::setw(20) << std::numeric_limits<uint32_t>::max() << std::endl;
		std::cout << "int64_t:        " << std::setw(20) << std::numeric_limits<int64_t>::min()
				  << " から " << std::setw(20) << std::numeric_limits<int64_t>::max() << std::endl;
		std::cout << "uint64_t:       " << std::setw(20) << std::numeric_limits<uint64_t>::min()
				  << " から " << std::setw(20) << std::numeric_limits<uint64_t>::max() << std::endl;

		// 各型のバイトサイズ
		std::cout << "\n【各型のバイトサイズ】" << std::endl;
		std::cout << "----------------------------------------" << std::endl;
		std::cout << "標準整数型:" << std::endl;
		std::cout << "char:           " << sizeof(char) << " バイト" << std::endl;
		std::cout << "short:          " << sizeof(short) << " バイト" << std::endl;
		std::cout << "int:            " << sizeof(int) << " バイト" << std::endl;
		std::cout << "long:           " << sizeof(long) << " バイト" << std::endl;
		std::cout << "long long:      " << sizeof(long long) << " バイト" << std::endl;

		std::cout << "\n固定幅整数型:" << std::endl;
		std::cout << "int8_t:         " << sizeof(int8_t) << " バイト" << std::endl;
		std::cout << "uint8_t:        " << sizeof(uint8_t) << " バイト" << std::endl;
		std::cout << "int16_t:        " << sizeof(int16_t) << " バイト" << std::endl;
		std::cout << "uint16_t:       " << sizeof(uint16_t) << " バイト" << std::endl;
		std::cout << "int32_t:        " << sizeof(int32_t) << " バイト" << std::endl;
		std::cout << "uint32_t:       " << sizeof(uint32_t) << " バイト" << std::endl;
		std::cout << "int64_t:        " << sizeof(int64_t) << " バイト" << std::endl;
		std::cout << "uint64_t:       " << sizeof(uint64_t) << " バイト" << std::endl;

		return 0;
	}

	else if (arg1 == "testfloat")
	{
		std::cout << "浮動小数点型の情報：" << std::endl;
		std::cout << "========================================" << std::endl;

		// 精度を高く設定
		std::cout << std::setprecision(std::numeric_limits<long double>::digits10 + 1);

		// 最大値・最小値
		std::cout << "【最大値と最小値】" << std::endl;
		std::cout << "----------------------------------------" << std::endl;
		std::cout << "float 最小値:      " << std::numeric_limits<float>::min() << std::endl;
		std::cout << "float 最大値:      " << std::numeric_limits<float>::max() << std::endl;
		std::cout << "double 最小値:     " << std::numeric_limits<double>::min() << std::endl;
		std::cout << "double 最大値:     " << std::numeric_limits<double>::max() << std::endl;
		std::cout << "long double 最小値: " << std::numeric_limits<long double>::min() << std::endl;
		std::cout << "long double 最大値: " << std::numeric_limits<long double>::max() << std::endl;

		// 精度
		std::cout << "\n【精度 (有効桁数)】" << std::endl;
		std::cout << "----------------------------------------" << std::endl;
		std::cout << "float:             " << std::numeric_limits<float>::digits10 << " 桁" << std::endl;
		std::cout << "double:            " << std::numeric_limits<double>::digits10 << " 桁" << std::endl;
		std::cout << "long double:       " << std::numeric_limits<long double>::digits10 << " 桁" << std::endl;

		// イプシロン（1.0より大きい最小の値と1.0の差）
		std::cout << "\n【イプシロン値】" << std::endl;
		std::cout << "----------------------------------------" << std::endl;
		std::cout << "float epsilon:      " << std::numeric_limits<float>::epsilon() << std::endl;
		std::cout << "double epsilon:     " << std::numeric_limits<double>::epsilon() << std::endl;
		std::cout << "long double epsilon: " << std::numeric_limits<long double>::epsilon() << std::endl;

		// サイズ
		std::cout << "\n【バイトサイズ】" << std::endl;
		std::cout << "----------------------------------------" << std::endl;
		std::cout << "float:             " << sizeof(float) << " バイト" << std::endl;
		std::cout << "double:            " << sizeof(double) << " バイト" << std::endl;
		std::cout << "long double:       " << sizeof(long double) << " バイト" << std::endl;

		// 特殊な値
		std::cout << "\n【特殊な値】" << std::endl;
		std::cout << "----------------------------------------" << std::endl;
		std::cout << "float 無限大:      " << std::numeric_limits<float>::infinity() << std::endl;
		std::cout << "float NaN:         " << std::numeric_limits<float>::quiet_NaN() << std::endl;
		std::cout << "double 無限大:     " << std::numeric_limits<double>::infinity() << std::endl;
		std::cout << "double NaN:        " << std::numeric_limits<double>::quiet_NaN() << std::endl;

		// 実際の計算例
		std::cout << "\n【計算例】" << std::endl;
		std::cout << "----------------------------------------" << std::endl;

		float f1 = 0.1f;
		float f2 = 0.2f;
		float f3 = f1 + f2;
		std::cout << "float:  0.1 + 0.2 = " << f3 << std::endl;

		double d1 = 0.1;
		double d2 = 0.2;
		double d3 = d1 + d2;
		std::cout << "double: 0.1 + 0.2 = " << d3 << std::endl;

		return 0;
	}

	else if (arg1 == "testdata")
	{
		try
		{
			// specify the path to the data file (data/jvd.dat)
			std::string dataFilePath = "data/jvd.dat";

			// check if the file exists
			std::ifstream file(dataFilePath);
			if (!file)
			{
				std::cerr << "Error: File not found: " << dataFilePath << std::endl;
				return 1;
			}

			// check the file size
			file.seekg(0, std::ios::end);
			std::streamsize fileSize = file.tellg();
			file.seekg(0, std::ios::beg);
			std::cout << "File size: " << fileSize << " bytes" << std::endl;

			// initialize record manager
			JVData::RecordManager recordManager;

			int processedCount = 0;
			int storedCount = 0;

			std::string line;
			while (std::getline(file, line))
			{
				std::unique_ptr<JVData::Record> record = JVData::RecordFactory::createRecord(line);
				if (record != nullptr)
				{
					processedCount++;
					// Try to add the record to the manager
					if (recordManager.addOrUpdateRecord(std::move(record)))
						storedCount++;
				}
			}

			std::cout << "Processed " << processedCount << " records." << std::endl;
			std::cout << "Stored " << storedCount << " records." << std::endl;

			file.close();

			// Add RecordManager functionality test
			std::cout << "\n=== RecordManager 機能テスト ===" << std::endl;
			std::cout << "合計レコード数: " << recordManager.size() << std::endl;

			// レコードタイプごとの統計を表示
			std::cout << "\n【レコードタイプ別集計】" << std::endl;
			std::unordered_map<std::string, int> recordTypeCounts;
			recordManager.forEach([&](const JVData::Record &record)
								  {
				std::string type = std::string(record.getRecordType().getValue());
				recordTypeCounts[type]++; });

			for (const auto &[type, count] : recordTypeCounts)
			{
				std::cout << type << ": " << count << " レコード" << std::endl;
			}

			// RAレコードの検索テスト
			std::cout << "\n【RAレコード検索テスト】" << std::endl;
			auto raRecords = recordManager.findRARecords();
			std::cout << "RAレコード数: " << raRecords.size() << std::endl;

			if (!raRecords.empty())
			{
				std::cout << "\n最初のRAレコードの詳細:" << std::endl;
				raRecords[0]->display();

				// 競馬場コードで検索
				std::string testKeibajo = "05"; // 東京競馬場
				auto tokyoRecords = recordManager.findRARecords(std::nullopt, testKeibajo);
				std::cout << "\n東京競馬場 (" << testKeibajo << ") のレース数: "
						  << tokyoRecords.size() << std::endl;
			}

			// SEレコードの検索テスト
			std::cout << "\n【SEレコード検索テスト】" << std::endl;
			auto seRecords = recordManager.findSERecords();
			std::cout << "SEレコード数: " << seRecords.size() << std::endl;

			if (!seRecords.empty())
			{
				std::cout << "\n最初のSEレコードの詳細:" << std::endl;
				seRecords[0]->display();

				// 1番人気の馬を探す
				std::cout << "\n特定の馬番(1番)の出走馬検索:" << std::endl;
				uint8_t testUmaBango = 1;
				auto uma1Records = recordManager.findSERecords(std::nullopt, std::nullopt, std::nullopt, testUmaBango);
				std::cout << "1番馬の出走回数: " << uma1Records.size() << std::endl;

				if (!uma1Records.empty())
				{
					std::cout << "例:" << std::endl;
					uma1Records[0]->display();
				}
			}

			// 血統登録番号に関する統計
			std::cout << "\n【血統登録番号統計】" << std::endl;
			std::unordered_map<uint64_t, int> kettoCount;

			// 血統登録番号ごとの出走回数をカウント
			recordManager.forEachOfType("SE", [&](const JVData::Record &record)
										{
                const auto* seRecord = dynamic_cast<const JVData::SERecord*>(&record);
                if (seRecord) {
                    kettoCount[seRecord->getKettoTorokuBango()]++;
                } });

			// ユニークな血統登録番号の数を表示
			std::cout << "ユニークな血統登録番号数: " << kettoCount.size() << std::endl;

			// 出走回数順にソートするため、vectorに変換
			std::vector<std::pair<uint64_t, int>> sortedKetto(kettoCount.begin(), kettoCount.end());
			std::sort(sortedKetto.begin(), sortedKetto.end(),
					  [](const auto &a, const auto &b)
					  { return a.second > b.second; });

			// トップ10の馬を表示
			std::cout << "\n【出走回数トップ10馬】" << std::endl;
			std::cout << "順位  血統登録番号  出走回数" << std::endl;
			std::cout << "----------------------------" << std::endl;

			int rank = 1;
			for (auto it = sortedKetto.begin();
				 it != sortedKetto.end() && rank <= 10;
				 ++it, ++rank)
			{
				std::cout << std::setw(4) << rank << "  "
						  << std::setw(12) << it->first << "  "
						  << std::setw(6) << it->second << "回" << std::endl;
			}

			return 0;
		}
		catch (const std::exception &e)
		{
			std::cerr << "Error: " << e.what() << std::endl;
			return 1;
		}
	}

	else
	{
		std::cerr << "Error: Invalid argument '" << arg1 << "'" << std::endl;
		return 1;
	}
}
