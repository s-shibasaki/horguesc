#include "RecordProcessor.h"

using namespace System;
using namespace Npgsql;



int RecordProcessor::ProcessHNRecord(array<Byte>^ record) {
	try {
		NpgsqlCommand^ command = gcnew NpgsqlCommand(nullptr, connection);

		command->Parameters->AddWithValue("@data_type", ByteSubstring(record, 2, 1));

		DateTime^ creationDate = DateTime::ParseExact(ByteSubstring(record, 3, 8), "yyyyMMdd", nullptr);
		command->Parameters->AddWithValue("@creation_date", creationDate);

		command->Parameters->AddWithValue("@hanshoku_toroku_bango", Int64::Parse(ByteSubstring(record, 11, 10)));
		command->Parameters->AddWithValue("@ketto_toroku_bango", Int64::Parse(ByteSubstring(record, 29, 10)));

		// 既存のレコードがあるか確認
		command->CommandText =
			"SELECT creation_date FROM hn WHERE hanshoku_toroku_bango = @hanshoku_toroku_bango";
		Object^ existingCreationDateObj = command->ExecuteScalar();

		// 既存のレコードがない場合は挿入して終了
		if (existingCreationDateObj == nullptr) {
			command->CommandText =
				"INSERT INTO hn (data_type, creation_date, hanshoku_toroku_bango, ketto_toroku_bango) "
				"VALUES (@data_type, @creation_date, @hanshoku_toroku_bango, @ketto_toroku_bango)";
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
			"UPDATE hn SET "
			"data_type = @data_type, "
			"creation_date = @creation_date, "
			"ketto_toroku_bango = @ketto_toroku_bango "
			"WHERE hanshoku_toroku_bango = @hanshoku_toroku_bango";
		command->ExecuteNonQuery();
		return PROCESS_SUCCESS;
	}
	catch (Exception^ ex) {
		Console::WriteLine();
		Console::WriteLine("Error processing HN record: {0}", ex->Message);
		return PROCESS_ERROR;
	}
}
