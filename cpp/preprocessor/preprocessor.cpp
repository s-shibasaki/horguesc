#include <iostream>
#include <fstream>
#include <limits>
#include <iomanip> // std::setprecision のために追加
#include "jvdata.h"
#include <algorithm>
#include <set>
#include <map>

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

            // Track record types for summary
            std::map<JVData::RecordType, int> recordTypeCounts;

            std::string line;
            while (std::getline(file, line))
            {
                std::unique_ptr<JVData::Record> record = JVData::RecordFactory::createRecord(line);
                if (record != nullptr)
                {
                    // Count record types using the RecordType object directly
                    recordTypeCounts[record->getRecordType()]++;

                    processedCount++;
                    // Try to add the record to the manager
                    if (recordManager.addOrUpdateRecord(std::move(record)))
                        storedCount++;
                }
            }

            file.close();

            // Print processing summary
            std::cout << "\n========== Data Processing Summary ==========\n";
            std::cout << "Processed: " << processedCount << " records total\n";
            std::cout << "Stored:    " << storedCount << " records\n";
            std::cout << "Added:     " << recordManager.getAddedCount() << " new records\n";
            std::cout << "Updated:   " << recordManager.getUpdatedCount() << " existing records\n";
            std::cout << "Skipped:   " << recordManager.getSkippedCount() << " older records\n";
            std::cout << "Errors:    " << recordManager.getErrorCount() << " records\n";

            // Print record type distribution
            std::cout << "\n========== Record Type Distribution ==========\n";
            for (const auto &[type, count] : recordTypeCounts)
            {
                // No need to create a new RecordType, use the map key directly
                std::cout << type.getValue() << " (" << type.getName() << "): " << count << " records\n";
            }

            // Demonstrate finding and displaying specific records
            std::cout << "\n========== Sample Record Queries ==========\n";

            // Sample 1: Display the first RA record found
            std::cout << "\n1. First RA record found:\n";
            bool foundRA = false;
            recordManager.findRARecords([&](const JVData::RARecord &record) -> bool
                                        {
                                            std::cout << "【レース詳細】\n";
                                            record.display();
                                            foundRA = true;
                                            return false; // Stop after first match
                                        });
            if (!foundRA)
                std::cout << "No RA records found.\n";

            // Sample 2: Find and display all RA records for a specific track (芝)
            std::cout << "\n2. 芝コースのレース:\n";
            JVData::TrackCode shibaTrackCode("11"); // 平地・芝・左回り
            int shibaCount = 0;
            recordManager.findRARecords([&](const JVData::RARecord &record) -> bool
                                        {
                                            shibaCount++;
                                            if (shibaCount <= 3)
                                            { // Display only first 3 matches
                                                std::cout << "【芝コースレース " << shibaCount << "】\n";
                                                record.display();
                                                std::cout << "----------------------------------------\n";
                                            }
                                            return true; // Continue finding all matches
                                        },
                                        std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, shibaTrackCode);
            std::cout << "芝コースのレース: 合計 " << shibaCount << " 件 (最初の3件のみ表示)\n";

            // Sample 3: Find all SE records for a specific race
            if (foundRA)
            {
                std::cout << "\n3. 特定のレースに出走する馬一覧:\n";
                bool raFound = false;
                recordManager.findRARecords([&](const JVData::RARecord &raRecord) -> bool
                                            {
                                                raFound = true;
                                                std::cout << "【レース情報】\n";
                                                raRecord.display();
                                                std::cout << "\n【出走馬一覧】\n";

                                                int horseCount = 0;
                                                recordManager.findSERecords([&](const JVData::SERecord &seRecord) -> bool
                                                                            {
                                                                                horseCount++;
                                                                                std::cout << "出走馬 " << static_cast<int>(seRecord.getUmaBango())
                                                                                          << " (枠番 " << static_cast<int>(seRecord.getWakuBango()) << "):\n";
                                                                                seRecord.display();
                                                                                std::cout << "----------------------------------------\n";
                                                                                return true; // Continue finding all horses
                                                                            },
                                                                            &raRecord);

                                                std::cout << "合計出走頭数: " << horseCount << " 頭\n";
                                                return false; // Stop after first race
                                            },
                                            std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt);

                if (!raFound)
                    std::cout << "No RA records found for the query.\n";
            }

            // Sample 4: Find races with specific distance range (middle distance: 1600m-2000m)
            std::cout << "\n4. 中距離 (1600m-2000m) のレース:\n";
            int middleDistanceCount = 0;
            recordManager.findRARecords([&](const JVData::RARecord &record) -> bool
                                        {
                                            middleDistanceCount++;
                                            if (middleDistanceCount <= 2)
                                            { // Display only first 2 matches
                                                std::cout << "【中距離レース " << middleDistanceCount << "】\n";
                                                record.display();
                                                std::cout << "----------------------------------------\n";
                                            }
                                            return true; // Continue finding all matches
                                        },
                                        std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::make_optional(std::pair<uint16_t, uint16_t>(1600, 2000)), std::nullopt);
            std::cout << "中距離レース: 合計 " << middleDistanceCount << " 件 (最初の2件のみ表示)\n";

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
