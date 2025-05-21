#include "RecordProcessor.h"

using namespace System;
using namespace System::Collections::Generic;
using namespace Npgsql;

int RecordProcessor::ProcessO1Record(array<Byte>^ record) {
    try {
        bool skip;
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

        // **************************************************
        // 単勝
        // **************************************************
        
		// レコードが存在するか確認する (念のため作成年月日が最新のものを取得する)
        command->Parameters->Clear();
        command->CommandText = "SELECT creation_date FROM tansho WHERE "
            "kaisai_date = @kaisai_date AND "
            "keibajo_code = @keibajo_code AND "
            "kaisai_kai = @kaisai_kai AND "
            "kaisai_nichime = @kaisai_nichime AND "
            "kyoso_bango = @kyoso_bango AND "
			"happyo_datetime = @happyo_datetime "
            "ORDER BY creation_date DESC LIMIT 1";
        command->Parameters->AddWithValue("@data_type", dataType);
        command->Parameters->AddWithValue("@creation_date", creationDate);
        command->Parameters->AddWithValue("@kaisai_date", kaisaiDate);
        command->Parameters->AddWithValue("@keibajo_code", keibajoCode);
        command->Parameters->AddWithValue("@kaisai_kai", kaisaiKai);
        command->Parameters->AddWithValue("@kaisai_nichime", kaisaiNichime);
        command->Parameters->AddWithValue("@kyoso_bango", kyosoBango);
        command->Parameters->AddWithValue("@happyo_datetime", happyoDateTime);
        existingDateObj = command->ExecuteScalar();

		// 存在する場合、作成年月日を確認する
        skip = false;
        if (existingDateObj != nullptr) {
            DateTime^ existingDate = Convert::ToDateTime(existingDateObj);

			// 作成年月日が、追加しようとしているものより新しい場合、処理をスキップ
            if (existingDate->CompareTo(creationDate) > 0)
                skip = true;

            // 追加しようとしているものと同じか、古い場合、削除する
            command->CommandText = "DELETE FROM tansho WHERE "
                "kaisai_date = @kaisai_date AND "
                "keibajo_code = @keibajo_code AND "
                "kaisai_kai = @kaisai_kai AND "
                "kaisai_nichime = @kaisai_nichime AND "
                "kyoso_bango = @kyoso_bango AND "
                "happyo_datetime = @happyo_datetime";
			command->ExecuteNonQuery();
        }

        // 存在しない場合、または既に存在するものを削除した場合、追加処理を行う
        if (!skip) {
            List<String^>^ tanshoValues = gcnew List<String^>();
            
            // Process each horse's odds data
            for (int i = 0; i < 28; i++) {
                Int16 umaban;
                try {
                    umaban = Int16::Parse(ByteSubstring(record, 43 + (i * 8), 2));
                }
                catch (...) {
                    continue;  // Skip if we can't parse a valid umaban (likely empty/padding)
                }
                if (umaban <= 0)
                    continue;  // Skip if umaban is 0
                
                String^ oddsValue;
                try {
                    Int32 odds = Int32::Parse(ByteSubstring(record, 43 + (i * 8) + 2, 4));
                    oddsValue = (odds != 0) ? odds.ToString() : "NULL";
                }
                catch (...) {
                    oddsValue = "NULL"; // Set to NULL if odds is not valid
                }
                
                String^ valueStr = String::Format("('{0}', '{1}', '{2}', '{3}', '{4}', '{5}', '{6}', '{7}', {8}, {9})",
                    dataType, creationDate->ToString("yyyy-MM-dd"), kaisaiDate->ToString("yyyy-MM-dd"), 
                    keibajoCode, kaisaiKai, kaisaiNichime, kyosoBango, 
                    happyoDateTime->ToString("yyyy-MM-dd HH:mm:ss"),
                    umaban, oddsValue);
                tanshoValues->Add(valueStr);
            }
            
            if (tanshoValues->Count > 0) {
                String^ valuesSql = String::Join(", ", tanshoValues);
                command->CommandText = "INSERT INTO tansho (data_type, creation_date, "
                    "kaisai_date, keibajo_code, kaisai_kai, kaisai_nichime, kyoso_bango, happyo_datetime, "
                    "umaban, odds) VALUES " + valuesSql;
                command->Parameters->Clear();
                command->ExecuteNonQuery();
            }
        }
        
        // **************************************************
        // 複勝
        // **************************************************

        // レコードが存在するか確認する (念のため作成年月日が最新のものを取得する)
        command->Parameters->Clear();
        command->CommandText = "SELECT creation_date FROM fukusho WHERE "
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
        skip = false;
        if (existingDateObj != nullptr) {
            DateTime^ existingDate = Convert::ToDateTime(existingDateObj);

            // 作成年月日が、追加しようとしているものより新しい場合、処理をスキップ
            if (existingDate->CompareTo(creationDate) > 0)
                skip = true;

            // 追加しようとしているものと同じか、古い場合、削除する
            command->CommandText = "DELETE FROM fukusho WHERE "
                "kaisai_date = @kaisai_date AND "
                "keibajo_code = @keibajo_code AND "
                "kaisai_kai = @kaisai_kai AND "
                "kaisai_nichime = @kaisai_nichime AND "
                "kyoso_bango = @kyoso_bango AND "
                "happyo_datetime = @happyo_datetime";
            command->ExecuteNonQuery();
        }

        // 存在しない場合、または既に存在するものを削除した場合、追加処理を行う
        if (!skip) {
            List<String^>^ fukushoValues = gcnew List<String^>();
            
            // Process each horse's odds data
            for (int i = 0; i < 28; i++) {
                Int16 umaban;
                try {
                    umaban = Int16::Parse(ByteSubstring(record, 267 + (i * 12), 2));
                }
                catch (...) {
                    continue;  // Skip if we can't parse a valid umaban (likely empty/padding)
                }
                if (umaban <= 0)
                    continue;  // Skip if umaban is 0
                
                String^ minOddsValue;
                try {
                    Int32 minOdds = Int32::Parse(ByteSubstring(record, 267 + (i * 12) + 2, 4));
                    minOddsValue = (minOdds != 0) ? minOdds.ToString() : "NULL";
                }
                catch (...) {
                    minOddsValue = "NULL"; // Set to NULL if odds is not valid
                }
                
                String^ maxOddsValue;
                try {
                    Int32 maxOdds = Int32::Parse(ByteSubstring(record, 267 + (i * 12) + 6, 4));
                    maxOddsValue = (maxOdds != 0) ? maxOdds.ToString() : "NULL";
                }
                catch (...) {
                    maxOddsValue = "NULL"; // Set to NULL if odds is not valid
                }
                
                String^ valueStr = String::Format("('{0}', '{1}', '{2}', '{3}', '{4}', '{5}', '{6}', '{7}', {8}, {9}, {10})",
                    dataType, creationDate->ToString("yyyy-MM-dd"), kaisaiDate->ToString("yyyy-MM-dd"), 
                    keibajoCode, kaisaiKai, kaisaiNichime, kyosoBango, 
                    happyoDateTime->ToString("yyyy-MM-dd HH:mm:ss"),
                    umaban, minOddsValue, maxOddsValue);
                fukushoValues->Add(valueStr);
            }
            
            if (fukushoValues->Count > 0) {
                String^ valuesSql = String::Join(", ", fukushoValues);
                command->CommandText = "INSERT INTO fukusho (data_type, creation_date, "
                    "kaisai_date, keibajo_code, kaisai_kai, kaisai_nichime, kyoso_bango, happyo_datetime, "
                    "umaban, min_odds, max_odds) VALUES " + valuesSql;
                command->Parameters->Clear();
                command->ExecuteNonQuery();
            }
        }

        // **************************************************
        // 枠連
        // **************************************************

        // レコードが存在するか確認する (念のため作成年月日が最新のものを取得する)
        command->Parameters->Clear();
        command->CommandText = "SELECT creation_date FROM wakuren WHERE "
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
        skip = false;
        if (existingDateObj != nullptr) {
            DateTime^ existingDate = Convert::ToDateTime(existingDateObj);

            // 作成年月日が、追加しようとしているものより新しい場合、処理をスキップ
            if (existingDate->CompareTo(creationDate) > 0)
                skip = true;

            // 追加しようとしているものと同じか、古い場合、削除する
            command->CommandText = "DELETE FROM wakuren WHERE "
                "kaisai_date = @kaisai_date AND "
                "keibajo_code = @keibajo_code AND "
                "kaisai_kai = @kaisai_kai AND "
                "kaisai_nichime = @kaisai_nichime AND "
                "kyoso_bango = @kyoso_bango AND "
                "happyo_datetime = @happyo_datetime";
            command->ExecuteNonQuery();
        }

        // 存在しない場合、または既に存在するものを削除した場合、追加処理を行う
        if (!skip) {
            List<String^>^ wakurenValues = gcnew List<String^>();
            
            // Process each horse's odds data
            for (int i = 0; i < 36; i++) {
                Int16 wakuban1;
                try {
                    wakuban1 = Int16::Parse(ByteSubstring(record, 603 + (i * 9), 1));
                }
                catch (...) {
                    continue;  // Skip if we can't parse a valid umaban (likely empty/padding)
                }
                if (wakuban1 <= 0)
                    continue;  // Skip if umaban is 0
                
                Int16 wakuban2;
                try {
                    wakuban2 = Int16::Parse(ByteSubstring(record, 603 + (i * 9) + 1, 1));
                }
                catch (...) {
                    continue;  // Skip if we can't parse a valid umaban (likely empty/padding)
                }
                if (wakuban2 <= 0)
                    continue;  // Skip if umaban is 0
                
                String^ oddsValue;
                try {
                    Int32 odds = Int32::Parse(ByteSubstring(record, 603 + (i * 9) + 2, 5));
                    oddsValue = (odds != 0) ? odds.ToString() : "NULL";
                }
                catch (...) {
                    oddsValue = "NULL"; // Set to NULL if odds is not valid
                }
                
                String^ valueStr = String::Format("('{0}', '{1}', '{2}', '{3}', '{4}', '{5}', '{6}', '{7}', {8}, {9}, {10})",
                    dataType, creationDate->ToString("yyyy-MM-dd"), kaisaiDate->ToString("yyyy-MM-dd"), 
                    keibajoCode, kaisaiKai, kaisaiNichime, kyosoBango, 
                    happyoDateTime->ToString("yyyy-MM-dd HH:mm:ss"),
                    wakuban1, wakuban2, oddsValue);
                wakurenValues->Add(valueStr);
            }
            
            if (wakurenValues->Count > 0) {
                String^ valuesSql = String::Join(", ", wakurenValues);
                command->CommandText = "INSERT INTO wakuren (data_type, creation_date, "
                    "kaisai_date, keibajo_code, kaisai_kai, kaisai_nichime, kyoso_bango, happyo_datetime, "
                    "wakuban_1, wakuban_2, odds) VALUES " + valuesSql;
                command->Parameters->Clear();
                command->ExecuteNonQuery();
            }
        }

        // **************************************************
        // 終了
        // **************************************************

        return PROCESS_SUCCESS;
    }
    catch (Exception^ ex) {
        Console::WriteLine();
        Console::WriteLine("Failed to process O1 record: {0}", ex->Message);
        return PROCESS_ERROR;
    }
}