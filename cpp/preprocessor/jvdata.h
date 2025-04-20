#pragma once

#include <string>
#include <array>
#include <sstream>
#include <chrono>
#include <memory>
#include <iomanip>
#include <unordered_map>
#include <functional>

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

        // Add not equal operator
        bool operator!=(const Code &other) const { return getValue() != other.getValue(); }
        bool operator!=(const std::string &str) const { return getValue() != str; }
        friend bool operator!=(const std::string &str, const Code &code) { return str != code.getValue(); }

        // Add less than operator
        bool operator<(const Code &other) const { return getValue() < other.getValue(); }
        bool operator<(const std::string &str) const { return getValue() < str; }
        friend bool operator<(const std::string &str, const Code &code) { return str < code.getValue(); }

        // Add greater than operator
        bool operator>(const Code &other) const { return getValue() > other.getValue(); }
        bool operator>(const std::string &str) const { return getValue() > str; }
        friend bool operator>(const std::string &str, const Code &code) { return str > code.getValue(); }

        // Add less than or equal operator
        bool operator<=(const Code &other) const { return getValue() <= other.getValue(); }
        bool operator<=(const std::string &str) const { return getValue() <= str; }
        friend bool operator<=(const std::string &str, const Code &code) { return str <= code.getValue(); }

        // Add greater than or equal operator
        bool operator>=(const Code &other) const { return getValue() >= other.getValue(); }
        bool operator>=(const std::string &str) const { return getValue() >= str; }
        friend bool operator>=(const std::string &str, const Code &code) { return str >= code.getValue(); }

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

    class TrackCode : public Code<2>
    {
    public:
        using Code<2>::Code;

        std::string getName() const override
        {
            const std::string_view code = getValue();

            if (code == "00")
                return "なし";
            if (code == "10")
                return "平地・芝・直線";
            if (code == "11")
                return "平地・芝・左回り";
            if (code == "12")
                return "平地・芝・左回り・外回り";
            if (code == "13")
                return "平地・芝・左回り・内－外回り";
            if (code == "14")
                return "平地・芝・左回り・外－内回り";
            if (code == "15")
                return "平地・芝・左回り・内２周";
            if (code == "16")
                return "平地・芝・左回り・外２周";
            if (code == "17")
                return "平地・芝・右回り";
            if (code == "18")
                return "平地・芝・右回り・外回り";
            if (code == "19")
                return "平地・芝・右回り・内－外回り";
            if (code == "20")
                return "平地・芝・右回り・外－内回り";
            if (code == "21")
                return "平地・芝・右回り・内２周";
            if (code == "22")
                return "平地・芝・右回り・外２周";
            if (code == "23")
                return "平地・ダート・左回り";
            if (code == "24")
                return "平地・ダート・右回り";
            if (code == "25")
                return "平地・ダート・左回り・内回り";
            if (code == "26")
                return "平地・ダート・右回り・外回り";
            if (code == "27")
                return "平地・サンド・左回り";
            if (code == "28")
                return "平地・サンド・右回り";
            if (code == "29")
                return "平地・ダート・直線";
            if (code == "51")
                return "障害・芝・襷";
            if (code == "52")
                return "障害・芝・ダート";
            if (code == "53")
                return "障害・芝・左";
            if (code == "54")
                return "障害・芝";
            if (code == "55")
                return "障害・芝・外回り";
            if (code == "56")
                return "障害・芝・外－内回り";
            if (code == "57")
                return "障害・芝・内－外回り";
            if (code == "58")
                return "障害・芝・内２周以上";
            if (code == "59")
                return "障害・芝・外２周以上";
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
        std::string_view getKey() const { return key; }

        // レコードの基本情報を文字列として返す
        virtual std::string toString() const
        {
            std::stringstream ss;
            ss << "レコード種別: " << recordType.getName() << " (" << recordType.getValue() << ")\n"
               << "データ区分: " << dataType->getName() << " (" << dataType->getValue() << ")\n"
               << "作成日: " << std::setw(4) << static_cast<int>(creationDate.year()) << "/"
               << std::setw(2) << std::setfill('0') << static_cast<unsigned>(creationDate.month()) << "/"
               << std::setw(2) << static_cast<unsigned>(creationDate.day()) << std::setfill(' ') << "\n"
               << "キー: " << key;
            return ss.str();
        }

        // 基本的な情報を表示する
        virtual void display() const { std::cout << toString() << std::endl; }
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
        const TrackCode trackCode;

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
              kyori(std::stoi(data.substr(697, 4))),
              trackCode(data.substr(705, 2)) {}

        // Getters for RA-specific properties
        const auto &getKaisaiDate() const { return kaisaiDate; }
        const auto &getKeibajoCode() const { return keibajoCode; }
        const auto &getKaisaiKai() const { return kaisaiKai; }
        const auto &getKaisaiNichime() const { return kaisaiNichime; }
        const auto &getKyosoBango() const { return kyosoBango; }
        const auto &getKyori() const { return kyori; } // Get the distance
        const auto &getTrackCode() const { return trackCode; }

        // RAレコードの情報を文字列として返す
        std::string toString() const override
        {
            std::stringstream ss;
            ss << Record::toString() << "\n"
               << "開催日: " << std::setw(4) << static_cast<int>(kaisaiDate.year()) << "/"
               << std::setw(2) << std::setfill('0') << static_cast<unsigned>(kaisaiDate.month()) << "/"
               << std::setw(2) << static_cast<unsigned>(kaisaiDate.day()) << std::setfill(' ') << "\n"
               << "競馬場: " << keibajoCode.getName() << " (" << keibajoCode.getValue() << ")\n"
               << "開催回数: " << static_cast<int>(kaisaiKai) << "\n"
               << "開催日目: " << static_cast<int>(kaisaiNichime) << "\n"
               << "競走番号: " << static_cast<int>(kyosoBango) << "\n"
               << "距離: " << kyori << "m\n"
               << "トラック: " << trackCode.getName() << " (" << trackCode.getValue() << ")";
            return ss.str();
        }
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

        // SEレコードの情報を文字列として返す
        std::string toString() const override
        {
            std::stringstream ss;
            ss << Record::toString() << "\n"
               << "開催日: " << std::setw(4) << static_cast<int>(kaisaiDate.year()) << "/"
               << std::setw(2) << std::setfill('0') << static_cast<unsigned>(kaisaiDate.month()) << "/"
               << std::setw(2) << static_cast<unsigned>(kaisaiDate.day()) << std::setfill(' ') << "\n"
               << "競馬場: " << keibajoCode.getName() << " (" << keibajoCode.getValue() << ")\n"
               << "開催回数: " << static_cast<int>(kaisaiKai) << "\n"
               << "開催日目: " << static_cast<int>(kaisaiNichime) << "\n"
               << "競走番号: " << static_cast<int>(kyosoBango) << "\n"
               << "枠番: " << static_cast<int>(wakuBango) << "\n"
               << "馬番: " << static_cast<int>(umaBango) << "\n"
               << "血統登録番号: " << kettoTorokuBango << "\n"
               << "馬齢: " << static_cast<int>(barei) << "\n"
               << "負担重量: " << getFutanJuryo() << "kg\n"
               << "ブリンカー: " << (blinker ? "あり" : "なし");
            return ss.str();
        }
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
            return std::make_unique<Record>(data); // Default to base class if type is unknown
        }
    };

    class RecordManager
    {
        // TODO: キーが同じでもレコードタイプが異なれば別のレコードとして扱うようにする
        // TODO: CreationDateが古い場合は上書きしないようにする
        // TODO: レコードを参照するためのメソッドを追加する
    private:
        // Use a composit key of record type + key
        using CompositeKey = std::pair<std::string, std::string>;

        // Hash function for the composite key
        struct CompositeKeyHash
        {
            std::size_t operator()(const CompositeKey &key) const
            {
                return std::hash<std::string>{}(key.first) ^
                       std::hash<std::string>{}(key.second);
            }
        };

        std::unordered_map<CompositeKey, std::unique_ptr<Record>, CompositeKeyHash> records;

        // Statistics counters
        size_t added_count = 0;   // New records added
        size_t updated_count = 0; // Records updated
        size_t skipped_count = 0; // Records skipped (older versions)
        size_t error_count = 0;   // Records with errors

    public:
        // Add or update a record in the manager
        bool addOrUpdateRecord(std::unique_ptr<Record> record)
        {
            if (!record)
            {
                error_count++;
                return false; // Invalid record
            }

            std::string_view keyView = record->getKey();
            if (keyView.empty())
            {
                error_count++;
                return false; // Invalid key
            }

            // Create composite key with record type and key
            std::string key(keyView);
            std::string recordTypeStr(record->getRecordType().getValue());
            CompositeKey compositeKey(recordTypeStr, key);

            // Check if we already have this record
            auto it = records.find(compositeKey);
            if (it != records.end())
            {
                // Check if existing record has a newer creation date
                auto existingDate = it->second->getCreationDate();
                auto newDate = record->getCreationDate();

                // Only update if the new record has a newer or equal date
                if (newDate < existingDate)
                {
                    skipped_count++; // Track skipped updates
                    return false;    // Don't update with older data
                }

                // Update existing record
                records[compositeKey] = std::move(record);
                updated_count++;
                return true;
            }

            // Add new record
            records[compositeKey] = std::move(record);
            added_count++;
            return true;
        }

        // Get methods for accessing records
        const Record *getRecord(const std::string &recordType, const std::string &key) const { return records.at({recordType, key}).get(); }

        // Get the number of records
        size_t size() const { return records.size(); }

        // Method to iterate over all records with early termination support
        void forEach(const std::function<bool(const Record &)> &callback) const
        {
            for (const auto &[key, record] : records)
            {
                if (!callback(*record))
                {
                    // Early termination if callback returns false
                    return;
                }
            }
        }

        // Generic search method that takes a predicate function and a callback
        // with early termination support
        template <typename T>
        void findRecords(const std::function<bool(const T &)> &callback,
                         const std::function<bool(const T &)> &predicate) const
        {
            forEach([&](const Record &record) -> bool
                    {
                        const auto *castRecord = dynamic_cast<const T *>(&record);
                        if (castRecord && predicate(*castRecord))
                        {
                            // Early terminate if callback returns false
                            return callback(*castRecord);
                        }
                        return true; // Continue iteration
                    });
        }

        void findRARecords(
            const std::function<bool(const RARecord &)> &callback,
            std::optional<std::chrono::year_month_day> kaisaiDate = std::nullopt,
            std::optional<std::pair<std::chrono::year_month_day, std::chrono::year_month_day>> kaisaiDateRange = std::nullopt,
            std::optional<KeibajoCode> keibajoCode = std::nullopt,
            std::optional<uint8_t> kaisaiKai = std::nullopt,
            std::optional<uint8_t> kaisaiNichime = std::nullopt,
            std::optional<uint8_t> kyosoBango = std::nullopt,
            std::optional<uint16_t> kyori = std::nullopt,
            std::optional<std::pair<uint16_t, uint16_t>> kyoriRange = std::nullopt,
            std::optional<TrackCode> trackCode = std::nullopt) const
        {
            findRecords<RARecord>(callback, [&](const RARecord &record) -> bool
                                  {
                    if (kaisaiDate && record.getKaisaiDate() != *kaisaiDate)
                        return false;
                    if (kaisaiDateRange) {
                        const auto& [startDate, endDate] = *kaisaiDateRange;
                        if (record.getKaisaiDate() < startDate || record.getKaisaiDate() > endDate)
                            return false;
                    }
                    if (keibajoCode && record.getKeibajoCode() != *keibajoCode)
                        return false;
                    if (kyosoBango && record.getKyosoBango() != *kyosoBango)
                        return false;
                    if (kyori && record.getKyori() != *kyori)
                        return false;
                    if (kyoriRange) {
                        const auto& [startKyori, endKyori] = *kyoriRange;
                        if (record.getKyori() < startKyori || record.getKyori() > endKyori)
                            return false;
                    }
                    if (trackCode && record.getTrackCode() != *trackCode)
                        return false;
                    return true; });
        }

        void findRARecords(const std::function<bool(const RARecord &)> &callback, const SERecord *seRecord) const
        {
            findRARecords(
                callback,
                seRecord->getKaisaiDate(),
                std::nullopt,
                seRecord->getKeibajoCode(),
                seRecord->getKaisaiKai(),
                seRecord->getKaisaiNichime(),
                seRecord->getKyosoBango());
        }

        void findSERecords(
            const std::function<bool(const SERecord &)> &callback,
            std::optional<std::chrono::year_month_day> kaisaiDate = std::nullopt,
            std::optional<std::pair<std::chrono::year_month_day, std::chrono::year_month_day>> kaisaiDateRange = std::nullopt,
            std::optional<KeibajoCode> keibajoCode = std::nullopt,
            std::optional<uint8_t> kaisaiKai = std::nullopt,
            std::optional<uint8_t> kaisaiNichime = std::nullopt,
            std::optional<uint8_t> kyosoBango = std::nullopt,
            std::optional<uint8_t> umaBango = std::nullopt,
            std::optional<uint8_t> kettoTorokuBango = std::nullopt) const
        {
            return findRecords<SERecord>(callback, [&](const SERecord &record)
                                         {
                if (kaisaiDate && record.getKaisaiDate() != *kaisaiDate)
                    return false;
                if (kaisaiDateRange) {
                    const auto& [startDate, endDate] = *kaisaiDateRange;
                    if (record.getKaisaiDate() < startDate || record.getKaisaiDate() > endDate)
                        return false;
                }
                if (keibajoCode && record.getKeibajoCode() != *keibajoCode)
                    return false;
                if (kyosoBango && record.getKyosoBango() != *kyosoBango)
                    return false;
                if (umaBango && record.getUmaBango() != *umaBango)
                    return false;
                if (kettoTorokuBango && record.getKettoTorokuBango() != *kettoTorokuBango)
                    return false;
                return true; });
        }

        void findSERecords(const std::function<bool(const SERecord &)> callback, const RARecord *raRecord) const
        {
            return findSERecords(
                callback,
                raRecord->getKaisaiDate(),
                std::nullopt,
                raRecord->getKeibajoCode(),
                raRecord->getKaisaiKai(),
                raRecord->getKaisaiNichime(),
                raRecord->getKyosoBango());
        }

        // Get statistics methods
        size_t getAddedCount() const { return added_count; }
        size_t getUpdatedCount() const { return updated_count; }
        size_t getSkippedCount() const { return skipped_count; }
        size_t getErrorCount() const { return error_count; }

        // Reset statistics
        void resetStatistics()
        {
            added_count = 0;
            updated_count = 0;
            skipped_count = 0;
            error_count = 0;
        }
    };
}