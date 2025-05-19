#include "RecordProcessor.h"

using namespace System;
using namespace Npgsql;

int RecordProcessor::ProcessWERecord(array<Byte>^ record) {
    try {
        NpgsqlCommand^ command = gcnew NpgsqlCommand(nullptr, connection);

        command->Parameters->AddWithValue("@data_type", ByteSubstring(record, 2, 1));

        DateTime^ creationDate = DateTime::ParseExact(ByteSubstring(record, 3, 8), "yyyyMMdd", nullptr);
        command->Parameters->AddWithValue("@creation_date", creationDate);

        String^ kaisaiDateStr = ByteSubstring(record, 11, 8);
        DateTime^ kaisaiDate = DateTime::ParseExact(kaisaiDateStr, "yyyyMMdd", nullptr);
        command->Parameters->AddWithValue("@kaisai_date", kaisaiDate);

        command->Parameters->AddWithValue("@keibajo_code", ByteSubstring(record, 19, 2));
        command->Parameters->AddWithValue("@kaisai_kai", Int16::Parse(ByteSubstring(record, 21, 2)));
        command->Parameters->AddWithValue("@kaisai_nichime", Int16::Parse(ByteSubstring(record, 23, 2)));
        
        // happyo_datetime を修正: 年はkaisaiDateから取得し、月日時分はByteSubstringから取得
        String^ happyoTimeStr = ByteSubstring(record, 25, 8); // MMddHHmm 形式
        String^ happyoYear = kaisaiDateStr->Substring(0, 4); // 年はkaisai_dateから
        String^ combinedDateTimeStr = String::Format("{0}{1}", happyoYear, happyoTimeStr);
        DateTime^ happyoDateTime = DateTime::ParseExact(combinedDateTimeStr, "yyyyMMddHHmm", nullptr);
        command->Parameters->AddWithValue("@happyo_datetime", happyoDateTime);
        
        command->Parameters->AddWithValue("@henko_shikibetsu", ByteSubstring(record, 33, 1));
        command->Parameters->AddWithValue("@tenko_code", ByteSubstring(record, 34, 1));
        command->Parameters->AddWithValue("@babajotai_code_shiba", ByteSubstring(record, 35, 1));
        command->Parameters->AddWithValue("@babajotai_code_dirt", ByteSubstring(record, 36, 1));

        // 既存のレコードがあるか確認
        command->CommandText = "SELECT creation_date FROM we WHERE "
            "kaisai_date = @kaisai_date AND "
            "keibajo_code = @keibajo_code AND "
            "kaisai_kai = @kaisai_kai AND "
            "kaisai_nichime = @kaisai_nichime AND "
            "happyo_datetime = @happyo_datetime AND " 
            "henko_shikibetsu = @henko_shikibetsu";
        Object^ existingCreationDateObj = command->ExecuteScalar();

        // 既存のレコードがない場合は挿入して終了
        if (existingCreationDateObj == nullptr) {
            command->CommandText = "INSERT INTO we (data_type, creation_date, "
                "kaisai_date, keibajo_code, kaisai_kai, kaisai_nichime, happyo_datetime, "
                "henko_shikibetsu, tenko_code, babajotai_code_shiba, babajotai_code_dirt) "
                "VALUES (@data_type, @creation_date, "
                "@kaisai_date, @keibajo_code, @kaisai_kai, @kaisai_nichime, @happyo_datetime, "
                "@henko_shikibetsu, @tenko_code, @babajotai_code_shiba, @babajotai_code_dirt)";
            command->ExecuteNonQuery();
            return PROCESS_SUCCESS;
        }

        // 既存のレコードがある場合は日付を確認
        DateTime^ existingCreationDate = Convert::ToDateTime(existingCreationDateObj);

        // 既存の日付の方が新しい場合は何もしないで終了
        if (existingCreationDate->CompareTo(creationDate) > 0)
            return PROCESS_SUCCESS;

        // 既存の日付の方が古い場合は更新
        command->CommandText = "UPDATE we SET "
            "data_type = @data_type, "
            "creation_date = @creation_date, "
            "tenko_code = @tenko_code, "
            "babajotai_code_shiba = @babajotai_code_shiba, "
            "babajotai_code_dirt = @babajotai_code_dirt "
            "WHERE kaisai_date = @kaisai_date AND "
            "keibajo_code = @keibajo_code AND "
            "kaisai_kai = @kaisai_kai AND "
            "kaisai_nichime = @kaisai_nichime AND "
            "happyo_datetime = @happyo_datetime AND " // スペースを追加
            "henko_shikibetsu = @henko_shikibetsu";
        command->ExecuteNonQuery();
        return PROCESS_SUCCESS;
    }
    catch (Exception^ ex) {
        Console::WriteLine();
        Console::WriteLine("Failed to process WE record: {0}", ex->Message);
        return PROCESS_ERROR;
    }
}