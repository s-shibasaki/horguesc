﻿#include "RecordProcessor.h"

using namespace System;
using namespace Npgsql;



int RecordProcessor::ProcessKSRecord(array<Byte>^ record) {
	try {
		NpgsqlCommand^ command = gcnew NpgsqlCommand(nullptr, connection);

		command->Parameters->AddWithValue("@data_type", ByteSubstring(record, 2, 1));

		DateTime^ creationDate = DateTime::ParseExact(ByteSubstring(record, 3, 8), "yyyyMMdd", nullptr);
		command->Parameters->AddWithValue("@creation_date", creationDate);

		command->Parameters->AddWithValue("@kishu_code", Int32::Parse(ByteSubstring(record, 11, 5)));
		
		// Extract birth date from position 33
		String^ birthDateStr = ByteSubstring(record, 33, 8);
		if (!String::IsNullOrEmpty(birthDateStr) && birthDateStr->Trim() != "00000000") {
			try {
				DateTime^ birthDate = DateTime::ParseExact(birthDateStr, "yyyyMMdd", nullptr);
				command->Parameters->AddWithValue("@birth_date", birthDate);
			}
			catch (Exception^) {
				// If birth date parsing fails, set to NULL
				command->Parameters->AddWithValue("@birth_date", DBNull::Value);
			}
		}
		else {
			command->Parameters->AddWithValue("@birth_date", DBNull::Value);
		}
		
		command->Parameters->AddWithValue("@seibetsu_kubun", ByteSubstring(record, 227, 1));

		// 既存のレコードがあるか確認
		command->CommandText =
			"SELECT creation_date FROM ks WHERE kishu_code = @kishu_code";
		Object^ existingCreationDateObj = command->ExecuteScalar();

		// 既存のレコードがない場合は挿入して終了
		if (existingCreationDateObj == nullptr) {
			command->CommandText =
				"INSERT INTO ks (data_type, creation_date, kishu_code, "
				"seibetsu_kubun, birth_date) "
				"VALUES (@data_type, @creation_date, @kishu_code, "
				"@seibetsu_kubun, @birth_date)";
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
			"UPDATE ks SET "
			"data_type = @data_type, "
			"creation_date = @creation_date, "
			"seibetsu_kubun = @seibetsu_kubun, "
			"birth_date = @birth_date "
			"WHERE kishu_code = @kishu_code";
		command->ExecuteNonQuery();
		return PROCESS_SUCCESS;
	}
	catch (Exception^ ex) {
		Console::WriteLine();
		Console::WriteLine("Error processing KS record: {0}", ex->Message);
		return PROCESS_ERROR;
	}
}
