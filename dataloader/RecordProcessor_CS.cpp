#include "RecordProcessor.h"

using namespace System;
using namespace Npgsql;



int RecordProcessor::ProcessCSRecord(array<Byte>^ record) {
	try {
		NpgsqlCommand^ command = gcnew NpgsqlCommand(nullptr, connection);

		command->Parameters->AddWithValue("@data_type", ByteSubstring(record, 2, 1));

		DateTime^ creationDate = DateTime::ParseExact(ByteSubstring(record, 3, 8), "yyyyMMdd", nullptr);
		command->Parameters->AddWithValue("@creation_date", creationDate);

		command->Parameters->AddWithValue("@keibajo_code", ByteSubstring(record, 11, 2));
		command->Parameters->AddWithValue("@kyori", Int16::Parse(ByteSubstring(record, 13, 4)));
		command->Parameters->AddWithValue("@track_code", ByteSubstring(record, 17, 2));
		command->Parameters->AddWithValue("@kaishu_date", DateTime::ParseExact(ByteSubstring(record, 19, 8), "yyyyMMdd", nullptr));
		command->Parameters->AddWithValue("@description", ByteSubstring(record, 27, 6800)->Trim()->Replace("\u3000", ""));

		// 既存のレコードがあるか確認
		command->CommandText =
			"SELECT creation_date FROM cs "
			"WHERE keibajo_code = @keibajo_code AND kyori = @kyori AND track_code = @track_code AND kaishu_date = @kaishu_date";
		Object^ existingCreationDateObj = command->ExecuteScalar();

		// 既存のレコードがない場合は挿入して終了
		if (existingCreationDateObj == nullptr) {
			command->CommandText =
				"INSERT INTO cs (data_type, creation_date, "
				"keibajo_code, kyori, track_code, kaishu_date, description) "
				"VALUES (@data_type, @creation_date, "
				"@keibajo_code, @kyori, @track_code, @kaishu_date, @description)";
			command->ExecuteNonQuery();
			return PROCESS_SUCCESS;
		}

		// 既存のレコードがある場合は日付を確認
		DateTime^ existingCreationDate = Convert::ToDateTime(existingCreationDateObj);

		// 既存の日付の方が新しい場合は何もしないで終了
		if (existingCreationDate->CompareTo(creationDate) > 0)
			return PROCESS_SUCCESS;

		// 既存の日付が古いか同じ場合は更新して終了
		command->CommandText =
			"UPDATE cs SET "
			"data_type = @data_type, "
			"creation_date = @creation_date, "
			"description = @description "
			"WHERE keibajo_code = @keibajo_code "
			"AND kyori = @kyori "
			"AND track_code = @track_code "
			"AND kaishu_date = @kaishu_date";
		command->ExecuteNonQuery();
		return PROCESS_SUCCESS;
	}
	catch (Exception^ ex) {
		Console::WriteLine();
		Console::WriteLine("Error processing CS record: {0}", ex->Message);
		return PROCESS_ERROR;
	}
}
