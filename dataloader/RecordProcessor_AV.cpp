#include "RecordProcessor.h"

using namespace System;
using namespace Npgsql;

int RecordProcessor::ProcessAVRecord(array<Byte>^ record) {
	try {
		NpgsqlCommand^ command = gcnew NpgsqlCommand(nullptr, connection);

		command->Parameters->AddWithValue("@data_type", ByteSubstring(record, 2, 1));

		DateTime^ creationDate = DateTime::ParseExact(ByteSubstring(record, 3, 8), "yyyyMMdd", nullptr);
		command->Parameters->AddWithValue("@creation_date", creationDate);

		command->Parameters->AddWithValue("@kaisai_date", DateTime::ParseExact(ByteSubstring(record, 11, 8), "yyyyMMdd", nullptr));
		command->Parameters->AddWithValue("@keibajo_code", ByteSubstring(record, 19, 2));
		command->Parameters->AddWithValue("@kaisai_kai", Int16::Parse(ByteSubstring(record, 21, 2)));
		command->Parameters->AddWithValue("@kaisai_nichime", Int16::Parse(ByteSubstring(record, 23, 2)));
		command->Parameters->AddWithValue("@kyoso_bango", Int16::Parse(ByteSubstring(record, 25, 2)));
		command->Parameters->AddWithValue("@umaban", Int16::Parse(ByteSubstring(record, 35, 2)));

		// 既存のレコードがあるか確認
		command->CommandText = "SELECT creation_date FROM av WHERE "
			"kaisai_date = @kaisai_date AND "
			"keibajo_code = @keibajo_code AND "
			"kaisai_kai = @kaisai_kai AND "
			"kaisai_nichime = @kaisai_nichime AND "
			"kyoso_bango = @kyoso_bango AND "
			"umaban = @umaban";
		Object^ existingCreationDateObj = command->ExecuteScalar();

		// 既存のレコードがない場合は挿入して終了
		if (existingCreationDateObj == nullptr) {
			command->CommandText = "INSERT INTO av (data_type, creation_date, "
				"kaisai_date, keibajo_code, kaisai_kai, kaisai_nichime, kyoso_bango, "
				"umaban) "
				"VALUES (@data_type, @creation_date, "
				"@kaisai_date, @keibajo_code, @kaisai_kai, @kaisai_nichime, @kyoso_bango, "
				"@umaban)";
			command->ExecuteNonQuery();
			return PROCESS_SUCCESS;
		}

		// 既存のレコードがある場合は日付を確認
		DateTime^ existingCreationDate = Convert::ToDateTime(existingCreationDateObj);

		// 既存の日付の方が新しい場合は何もしないで終了
		if (existingCreationDate->CompareTo(creationDate) > 0)
			return PROCESS_SUCCESS;

		// 既存の日付の方が古い場合は更新
		command->CommandText = "UPDATE av SET "
			"data_type = @data_type, "
			"creation_date = @creation_date "
			"WHERE kaisai_date = @kaisai_date AND "
			"keibajo_code = @keibajo_code AND "
			"kaisai_kai = @kaisai_kai AND "
			"kaisai_nichime = @kaisai_nichime AND "
			"kyoso_bango = @kyoso_bango AND "
			"umaban = @umaban";
		command->ExecuteNonQuery();
		return PROCESS_SUCCESS;
	}
	catch (Exception^ ex) {
		Console::WriteLine();
		Console::WriteLine("Failed to process AV record: {0}", ex->Message);
		return PROCESS_ERROR;
	}
}