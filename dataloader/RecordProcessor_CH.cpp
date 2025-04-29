#include "RecordProcessor.h"

using namespace System;
using namespace Npgsql;



int RecordProcessor::ProcessCHRecord(array<Byte>^ record) {
	try {
		NpgsqlCommand^ command = gcnew NpgsqlCommand(nullptr, connection);

		command->Parameters->AddWithValue("@data_type", ByteSubstring(record, 2, 1));

		DateTime^ creationDate = DateTime::ParseExact(ByteSubstring(record, 3, 8), "yyyyMMdd", nullptr);
		command->Parameters->AddWithValue("@creation_date", creationDate);

		command->Parameters->AddWithValue("@chokyoshi_code", Int32::Parse(ByteSubstring(record, 11, 5)));

		// 既存のレコードがあるか確認
		command->CommandText =
			"SELECT creation_date FROM ch WHERE chokyoshi_code = @chokyoshi_code";
		Object^ existingCreationDateObj = command->ExecuteScalar();

		// 既存のレコードがない場合は挿入して終了
		if (existingCreationDateObj == nullptr) {
			command->CommandText =
				"INSERT INTO ch (data_type, creation_date, chokyoshi_code) "
				"VALUES (@data_type, @creation_date, @chokyoshi_code)";
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
			"UPDATE ch SET "
			"data_type = @data_type, "
			"creation_date = @creation_date, "
			"WHERE chokyoshi_code = @chokyoshi_code";
		command->ExecuteNonQuery();
		return PROCESS_SUCCESS;
	}
	catch (Exception^ ex) {
		Console::WriteLine();
		Console::WriteLine("Error processing CH record: {0}", ex->Message);
		return PROCESS_ERROR;
	}
}
