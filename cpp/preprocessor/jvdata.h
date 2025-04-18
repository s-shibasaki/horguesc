#pragma once

#include <string>
#include <array>
#include <sstream>
#include <chrono>
#include <memory>
#include <iomanip>
#include <unordered_map>

namespace JVData
{
    template <size_t N>
    class Code : std::array<char, N + 1>
    {
    public:
        Code(const std::string &data)
        {
            if (data.size() != N)
                throw std::invalid_argument("Input string must be exactly " + std::to_string(N) + " characters");
            std::copy_n(data.begin(), N, this->begin());
            (*this)[N] = '\0';
        }

        // Get the value as a string
        std::string_view getValue() const { return std::string_view(this->data(), N); }
        virtual std::string getName() const = 0;

        // Overload the operator== for comparison
        bool operator==(const Code &other) const { return getValue() == other.getValue(); }
        bool operator==(const std::string &str) const { return getValue() == str; }
        friend bool operator==(const std::string &str, const Code &code) { return str == code.getValue(); }

        // Overload the operator<< for output
        friend std::ostream &operator<<(std::ostream &os, const Code &code) { return os << code.getValue(); }
    };

    class RecordType : public Code<2>
    {
    public:
        using Code<2>::Code; // Inherit constructor from Code<2>

        std::string getName() const override
        {
            const std::string_view code = getValue();
            if (code == "TK")
                return "特別登録馬";
            if (code == "RA")
                return "レース詳細";
            if (code == "SE")
                return "馬毎レース詳細";
            if (code == "HR")
                return "払戻";
            if (code == "H1")
                return "票数1";
            if (code == "H6")
                return "票数6 (3連単)";
            if (code == "O1")
                return "オッズ1 (単複枠)";
            if (code == "O2")
                return "オッズ2 (馬連)";
            if (code == "O3")
                return "オッズ3 (ワイド)";
            if (code == "O4")
                return "オッズ4 (馬単)";
            if (code == "O5")
                return "オッズ5 (3連複)";
            if (code == "O6")
                return "オッズ6 (3連単)";
            if (code == "UM")
                return "競走馬マスタ";
            if (code == "KS")
                return "騎手マスタ";
            if (code == "CH")
                return "調教師マスタ";
            if (code == "BR")
                return "生産者マスタ";
            if (code == "BN")
                return "馬主マスタ";
            if (code == "HN")
                return "繁殖馬マスタ";
            if (code == "SK")
                return "産駒マスタ";
            if (code == "CK")
                return "出走別着度数";
            if (code == "RC")
                return "レコードマスタ";
            if (code == "HC")
                return "坂路調教";
            if (code == "HS")
                return "競走馬市場取引価格";
            if (code == "HY")
                return "馬名の意味由来";
            if (code == "YS")
                return "開催スケジュール";
            if (code == "BT")
                return "系統情報";
            if (code == "CS")
                return "コース情報";
            if (code == "DM")
                return "タイム型データマイニング予想";
            if (code == "TM")
                return "対戦型データマイニング予想";
            if (code == "WF")
                return "重勝式 (WIN5)";
            if (code == "JG")
                return "競走馬除外情報";
            if (code == "WC")
                return "ウッドチップ調教";
            if (code == "WH")
                return "馬体重";
            if (code == "WE")
                return "天候馬場状態";
            if (code == "AV")
                return "出走取消・競争除外";
            if (code == "JC")
                return "騎手変更";
            if (code == "TC")
                return "発走時刻変更";
            if (code == "CC")
                return "コース変更";
            return "不明";
        }
    };

    class DataType : public Code<1>
    {
    public:
        using Code<1>::Code; // Inherit constructor from Code<1>

        virtual std::string getName() const override { return "不明"; }
    };

    class KeibajoCode : public Code<2>
    {
    public:
        using Code<2>::Code; // Inherit constructor from Code<2>

        std::string getName() const override
        {
            const std::string_view code = getValue();
            if (code == "00")
                return "なし";
            if (code == "01")
                return "札幌";
            if (code == "02")
                return "函館";
            if (code == "03")
                return "福島";
            if (code == "04")
                return "新潟";
            if (code == "05")
                return "東京";
            if (code == "06")
                return "中山";
            if (code == "07")
                return "中京";
            if (code == "08")
                return "京都";
            if (code == "09")
                return "阪神";
            if (code == "10")
                return "小倉";
            if (code >= "30" && code <= "61")
                return "地方";
            if (code >= "A0" && code <= "N6")
                return "海外";
            return "不明";
        }
    };

    // Record class with nested DataType implementation
    class Record
    {
    private:
        const RecordType recordType;                      // レコード種別ID
        const std::unique_ptr<JVData::DataType> dataType; // データ区分
        const std::chrono::year_month_day creationDate;   // データ作成年月日
        const std::string key;

    public:
        // Constructor for Record - 派生クラスがdataTypeを指定できるようにする
        Record(const std::string &data, std::unique_ptr<JVData::DataType> dataType = nullptr, std::string key = "")
            : recordType(data.substr(0, 2)),
              dataType(dataType ? std::move(dataType) : std::make_unique<DataType>(data.substr(2, 1))),
              creationDate(std::chrono::year(std::stoi(data.substr(3, 4))),
                           std::chrono::month(std::stoi(data.substr(7, 2))),
                           std::chrono::day(std::stoi(data.substr(9, 2)))),
              key(std::move(key)) {}

        // Destructor
        virtual ~Record() = default; // Virtual destructor for proper cleanup of derived classes

        // Getters for Record properties
        const auto &getRecordType() const { return recordType; }
        const auto &getDataType() const { return *dataType; } // Return a reference to DataType
        const auto &getCreationDate() const { return creationDate; }
        virtual std::string_view getKey() { return key; }
    };

    class RARecord : public Record
    {
    public:
        // DataType class for RARecord
        class DataType : public JVData::DataType
        {
        public:
            using JVData::DataType::DataType; // Inherit constructor from DataType

            std::string getName() const override
            {
                const std::string_view code = getValue();
                if (code == "1")
                    return "出走馬名表 (木曜)";
                if (code == "2")
                    return "出馬表 (金・土曜)";
                if (code == "3")
                    return "速報成績 (3着まで確定)";
                if (code == "4")
                    return "速報成績 (5着まで確定)";
                if (code == "5")
                    return "速報成績 (全馬着順確定)";
                if (code == "6")
                    return "速報成績 (全馬着順 + コーナ通過順)";
                if (code == "7")
                    return "成績 (月曜)";
                if (code == "A")
                    return "地方競馬";
                if (code == "B")
                    return "海外国際レース";
                if (code == "9")
                    return "レース中止";
                if (code == "0")
                    return "該当レコード削除 (提供ミスなどの理由による)";
                return "不明";
            }
        };

    private:
        const std::chrono::year_month_day kaisaiDate;
        const KeibajoCode keibajoCode;
        const uint8_t kaisaiKai;
        const uint8_t kaisaiNichime;
        const uint8_t kyosoBango;
        const uint16_t kyori;

    public:
        // Constructor
        RARecord(const std::string &data)
            : Record(data, std::make_unique<DataType>(data.substr(2, 1)), data.substr(11, 16)),
              kaisaiDate(std::chrono::year(std::stoi(data.substr(11, 4))),
                         std::chrono::month(std::stoi(data.substr(15, 2))),
                         std::chrono::day(std::stoi(data.substr(17, 2)))),
              keibajoCode(data.substr(19, 2)),
              kaisaiKai(std::stoi(data.substr(21, 2))),
              kaisaiNichime(std::stoi(data.substr(23, 2))),
              kyosoBango(std::stoi(data.substr(25, 2))),
              kyori(std::stoi(data.substr(697, 4))) {}

        // Getters for RA-specific properties
        const auto &getKaisaiDate() const { return kaisaiDate; }
        const auto &getKeibajoCode() const { return keibajoCode; }
        const auto &getKaisaiKai() const { return kaisaiKai; }
        const auto &getKaisaiNichime() const { return kaisaiNichime; }
        const auto &getKyosoBango() const { return kyosoBango; }
        const auto &getKyori() const { return kyori; } // Get the distance
    };

    class SERecord : public Record
    {
    private:
        const std::chrono::year_month_day kaisaiDate;
        const KeibajoCode keibajoCode;
        const uint8_t kaisaiKai;
        const uint8_t kaisaiNichime;
        const uint8_t kyosoBango;
        const uint8_t wakuBango;
        const uint8_t umaBango;
        const uint64_t kettoTorokuBango;
        const uint8_t barei;
        const uint16_t futanJuryo;
        const bool blinker;

    public:
        // Constructor
        SERecord(const std::string &data)
            : Record(data, std::make_unique<RARecord::DataType>(data.substr(2, 1)), data.substr(11, 16) + data.substr(28, 12)),
              kaisaiDate(std::chrono::year(std::stoi(data.substr(11, 4))),
                         std::chrono::month(std::stoi(data.substr(15, 2))),
                         std::chrono::day(std::stoi(data.substr(17, 2)))),
              keibajoCode(data.substr(19, 2)),
              kaisaiKai(std::stoi(data.substr(21, 2))),
              kaisaiNichime(std::stoi(data.substr(23, 2))),
              kyosoBango(std::stoi(data.substr(25, 2))),
              wakuBango(std::stoi(data.substr(27, 1))),
              umaBango(std::stoi(data.substr(28, 2))),
              kettoTorokuBango(std::stoull(data.substr(30, 10))),
              barei(std::stoi(data.substr(82, 2))),
              futanJuryo(std::stoi(data.substr(288, 3))),
              blinker((data[294] == '1')) {}

        // Getters for SE-specific properties
        const auto &getKaisaiDate() const { return kaisaiDate; }
        const auto &getKeibajoCode() const { return keibajoCode; }
        const auto &getKaisaiKai() const { return kaisaiKai; }
        const auto &getKaisaiNichime() const { return kaisaiNichime; }
        const auto &getKyosoBango() const { return kyosoBango; }
        const auto &getWakuBango() const { return wakuBango; }
        const auto &getUmaBango() const { return umaBango; }
        const auto &getKettoTorokuBango() const { return kettoTorokuBango; }
        const auto &getBarei() const { return barei; }
        float getFutanJuryo() const { return futanJuryo / 10.0f; }
        const auto &getBlinker() const { return blinker; }
    };

    // Factory class for creating Record objects based on record type
    class RecordFactory
    {
    public:
        static std::unique_ptr<Record> createRecord(const std::string &data)
        {
            if (data.length() < 2)
                throw std::invalid_argument("Data too short to determine record type");

            std::string recordType = data.substr(0, 2);

            // Create the appropriate record based on record type
            if (recordType == "RA")
                return std::make_unique<RARecord>(data);
            else if (recordType == "SE")
                return std::make_unique<SERecord>(data);
            else
                return std::make_unique<Record>(data); // Default to base class if type is unknown
        }
    };

    class RecordManager
    {
        // TODO: キーが同じでもレコードタイプが異なれば別のレコードとして扱うようにする
        // TODO: CreationDateが古い場合は上書きしないようにする
        // TODO: レコードを参照するためのメソッドを追加する
    private:
        std::unordered_map<std::string_view, std::unique_ptr<Record>> recordsByKey;

    public:
        bool addOrUpdateRecord(std::unique_ptr<Record> record)
        {
            if (!record)
                return false; // Invalid record

            std::string_view key = record->getKey();
            if (key.empty())
                return false; // Invalid key

            recordsByKey[key] = std::move(record); // Add or update the record
            return true;
        }
    };
}