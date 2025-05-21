#include "RecordProcessor.h"

using namespace System;
using namespace System::Collections::Generic;
using namespace Npgsql;

int RecordProcessor::ProcessO5Record(array<Byte>^ record) {
    try {
        NpgsqlCommand^ command = gcnew NpgsqlCommand(nullptr, connection);
        Object^ existingDateObj;

        DateTime^ creationDate = DateTime::ParseExact(ByteSubstring(record, 3, 8), "yyyyMMdd", nullptr);
        String^ dataType = ByteSubstring(record, 2, 1);
        String^ kaisaiDateStr = ByteSubstring(record, 11, 8);
        DateTime^ kaisaiDate = DateTime::ParseExact(kaisaiDateStr, "yyyyMMdd", nullptr);
        String^ keibajoCode = ByteSubstring(record, 19, 2);
        Int16 kaisaiKai = Int16::Parse(ByteSubstring(record, 21, 2));
        Int16 kaisaiNichime = Int16::Parse(ByteSubstring(record, 23, 2));
        Int16 kyosoBango = Int16::Parse(ByteSubstring(record, 25, 2));

        // happyo_datetime 処理: '00000000'の場合は特別処理
        String^ happyoTimeStr = ByteSubstring(record, 27, 8); // MMddHHmm 形式
        DateTime^ happyoDateTime;
        if (happyoTimeStr == "00000000") {
            // '00000000'の場合はデータ作成年月日の0時0分を使用
            happyoDateTime = DateTime(creationDate->Year, creationDate->Month, creationDate->Day, 0, 0, 0);
        }
        else {
            // 通常処理: 年はkaisai_dateから取得し、月日時分はByteSubstringから取得
            String^ happyoYear = kaisaiDateStr->Substring(0, 4);
            String^ combinedDateTimeStr = String::Format("{0}{1}", happyoYear, happyoTimeStr);
            happyoDateTime = DateTime::ParseExact(combinedDateTimeStr, "yyyyMMddHHmm", nullptr);
        }

        command->Parameters->Clear();
        command->CommandText = "SELECT creation_date FROM sanrenpuku WHERE "
            "kaisai_date = @kaisai_date AND "
            "keibajo_code = @keibajo_code AND "
            "kaisai_kai = @kaisai_kai AND "
            "kaisai_nichime = @kaisai_nichime AND "
            "kyoso_bango = @kyoso_bango AND "
            "happyo_datetime = @happyo_datetime "
            "ORDER BY creation_date DESC LIMIT 1";
        command->Parameters->AddWithValue("@kaisai_date", kaisaiDate);
        command->Parameters->AddWithValue("@keibajo_code", keibajoCode);
        command->Parameters->AddWithValue("@kaisai_kai", kaisaiKai);
        command->Parameters->AddWithValue("@kaisai_nichime", kaisaiNichime);
        command->Parameters->AddWithValue("@kyoso_bango", kyosoBango);
        command->Parameters->AddWithValue("@happyo_datetime", happyoDateTime);
        existingDateObj = command->ExecuteScalar();

        // 存在する場合、作成年月日を確認する
        if (existingDateObj != nullptr) {
            DateTime^ existingDate = Convert::ToDateTime(existingDateObj);

            // 作成年月日が、追加しようとしているものより新しい場合、処理をスキップ
            if (existingDate->CompareTo(creationDate) > 0)
                return PROCESS_SUCCESS;

            // 追加しようとしているものと同じか、古い場合、削除する
            command->CommandText = "DELETE FROM sanrenpuku WHERE "
                "kaisai_date = @kaisai_date AND "
                "keibajo_code = @keibajo_code AND "
                "kaisai_kai = @kaisai_kai AND "
                "kaisai_nichime = @kaisai_nichime AND "
                "kyoso_bango = @kyoso_bango AND "
                "happyo_datetime = @happyo_datetime";
            command->ExecuteNonQuery();
        }

        // 存在しない場合、または既に存在するものを削除した場合、追加処理を行う
        List<String^>^ batchValues = gcnew List<String^>();
        
        // Process each horse's odds data
        for (int i = 0; i < 816; i++) {
            Int16 umaban1;
            try {
                umaban1 = Int16::Parse(ByteSubstring(record, 40 + (i * 15), 2));
            }
            catch (...) {
                continue;  // Skip if we can't parse a valid umaban (likely empty/padding)
            }
            if (umaban1 <= 0)
                continue;  // Skip if umaban is 0

            Int16 umaban2;
            try {
                umaban2 = Int16::Parse(ByteSubstring(record, 40 + (i * 15) + 2, 2));
            }
            catch (...) {
                continue;  // Skip if we can't parse a valid umaban (likely empty/padding)
            }
            if (umaban2 <= 0)
                continue;  // Skip if umaban is 0

            Int16 umaban3;
            try {
                umaban3 = Int16::Parse(ByteSubstring(record, 40 + (i * 15) + 4, 2));
            }
            catch (...) {
                continue;  // Skip if we can't parse a valid umaban (likely empty/padding)
            }
            if (umaban3 <= 0)
                continue;  // Skip if umaban is 0

            String^ oddsValue;
            try {
                Int32 odds = Int32::Parse(ByteSubstring(record, 40 + (i * 15) + 6, 6));
                oddsValue = (odds != 0) ? odds.ToString() : "NULL";
            }
            catch (...) {
                oddsValue = "NULL"; // Set to NULL if odds is not valid
            }
            
            String^ valueStr = String::Format("('{0}', '{1}', '{2}', '{3}', '{4}', '{5}', '{6}', '{7}', {8}, {9}, {10}, {11})",
                dataType, creationDate->ToString("yyyy-MM-dd"), kaisaiDate->ToString("yyyy-MM-dd"), 
                keibajoCode, kaisaiKai, kaisaiNichime, kyosoBango, 
                happyoDateTime->ToString("yyyy-MM-dd HH:mm:ss"),
                umaban1, umaban2, umaban3, oddsValue);
            batchValues->Add(valueStr);
        }
        
        if (batchValues->Count > 0) {
            String^ valuesSql = String::Join(", ", batchValues);
            command->CommandText = "INSERT INTO sanrenpuku (data_type, creation_date, "
                "kaisai_date, keibajo_code, kaisai_kai, kaisai_nichime, kyoso_bango, happyo_datetime, "
                "umaban_1, umaban_2, umaban_3, odds) VALUES " + valuesSql;
            command->Parameters->Clear();
            command->ExecuteNonQuery();
        }

        return PROCESS_SUCCESS;
    }
    catch (Exception^ ex) {
        Console::WriteLine();
        Console::WriteLine("Failed to process O5 record: {0}", ex->Message);
        return PROCESS_ERROR;
    }
}