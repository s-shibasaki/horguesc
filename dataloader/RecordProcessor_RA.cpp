#include "RecordProcessor.h"

using namespace System;
using namespace Npgsql;

int RecordProcessor::ProcessRARecord(array<Byte>^ record) {
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
		command->Parameters->AddWithValue("@kyori", Int16::Parse(ByteSubstring(record, 697, 4)));

		String^ trackCode = ByteSubstring(record, 705, 2);
		command->Parameters->AddWithValue("@track_code", trackCode);

		command->Parameters->AddWithValue("@course_kubun", ByteSubstring(record, 709, 2));
		command->Parameters->AddWithValue("@tenko_code", ByteSubstring(record, 887, 1));

		// Add hasso_time parameter from bytes 873-876 (HHmm format)
		String^ hassoTimeStr = ByteSubstring(record, 873, 4);
		if (!String::IsNullOrEmpty(hassoTimeStr) && hassoTimeStr != "0000") {
			TimeSpan hassoTime = TimeSpan(Int32::Parse(hassoTimeStr->Substring(0, 2)), 
										Int32::Parse(hassoTimeStr->Substring(2, 2)), 0);
			command->Parameters->AddWithValue("@hasso_time", hassoTime);
		} else {
			command->Parameters->AddWithValue("@hasso_time", DBNull::Value);
		}

		Int16 trackCodeInt = Int16::Parse(trackCode);
		String^ babajotaiCode = "0";
		if ((10 <= trackCodeInt && trackCodeInt <= 22) || trackCodeInt == 51 || (53 <= trackCodeInt && trackCodeInt <= 59))
			babajotaiCode = ByteSubstring(record, 888, 1);
		else if ((23 <= trackCodeInt && trackCodeInt <= 29) || trackCodeInt == 52)
			babajotaiCode = ByteSubstring(record, 889, 1);
		command->Parameters->AddWithValue("@babajotai_code", babajotaiCode);

		// 既存のレコードがあるか確認
		command->CommandText = "SELECT creation_date FROM ra WHERE "
			"kaisai_date = @kaisai_date AND "
			"keibajo_code = @keibajo_code AND "
			"kaisai_kai = @kaisai_kai AND "
			"kaisai_nichime = @kaisai_nichime AND "
			"kyoso_bango = @kyoso_bango";
		Object^ existingCreationDateObj = command->ExecuteScalar();

		// 既存のレコードがない場合は挿入して終了
		if (existingCreationDateObj == nullptr) {
			// For INSERT statement
			command->CommandText = "INSERT INTO ra (data_type, creation_date, "
				"kaisai_date, keibajo_code, kaisai_kai, kaisai_nichime, kyoso_bango, "
				"kyori, track_code, course_kubun, tenko_code, babajotai_code, hasso_time) "
				"VALUES (@data_type, @creation_date, "
				"@kaisai_date, @keibajo_code, @kaisai_kai, @kaisai_nichime, @kyoso_bango, "
				"@kyori, @track_code, @course_kubun, @tenko_code, @babajotai_code, @hasso_time)";
			command->ExecuteNonQuery();
			return PROCESS_SUCCESS;
		}

		// 既存のレコードがある場合は日付を確認
		DateTime^ existingCreationDate = Convert::ToDateTime(existingCreationDateObj);

		// 既存の日付の方が新しい場合は何もしないで終了
		if (existingCreationDate->CompareTo(creationDate) > 0)
			return PROCESS_SUCCESS;

		// 既存の日付が古いか同じ場合は更新して終了
		// For UPDATE statement
		command->CommandText = "UPDATE ra SET "
			"data_type = @data_type, "
			"creation_date = @creation_date, "
			"kyori = @kyori, "
			"track_code = @track_code, "
			"course_kubun = @course_kubun, "
			"tenko_code = @tenko_code, "
			"babajotai_code = @babajotai_code, "
			"hasso_time = @hasso_time "
			"WHERE kaisai_date = @kaisai_date "
			"AND keibajo_code = @keibajo_code "
			"AND kaisai_kai = @kaisai_kai "
			"AND kaisai_nichime = @kaisai_nichime "
			"AND kyoso_bango = @kyoso_bango";
		command->ExecuteNonQuery();
		return PROCESS_SUCCESS;
	}
	catch (Exception^ ex) {
		Console::WriteLine();
		Console::WriteLine("Error processing RA record: {0}", ex->Message);
		return PROCESS_ERROR;
	}
}
