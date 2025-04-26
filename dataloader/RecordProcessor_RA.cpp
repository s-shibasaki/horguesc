#include "RecordProcessor.h"

using namespace System;
using namespace Npgsql;

int RecordProcessor::ProcessRARecord(String^ record) {
	try {
		NpgsqlCommand^ command = gcnew NpgsqlCommand(nullptr, connection);

		command->Parameters->AddWithValue("@data_type", record->Substring(2, 1));

		DateTime^ creationDate = DateTime::ParseExact(record->Substring(3, 8), "yyyyMMdd", nullptr);
		command->Parameters->AddWithValue("@creation_date", creationDate);

		command->Parameters->AddWithValue("@kaisai_date", DateTime::ParseExact(record->Substring(11, 8), "yyyyMMdd", nullptr));
		command->Parameters->AddWithValue("@keibajo_code", record->Substring(19, 2));
		command->Parameters->AddWithValue("@kyoso_bango", Int16::Parse(record->Substring(25, 2)));

		Int16 kyori = Int16::Parse(record->Substring(697, 4));
		if (kyori > 0)
			command->Parameters->AddWithValue("@kyori", kyori);
		else
			command->Parameters->AddWithValue("@kyori", DBNull::Value);

		String^ trackCode = record->Substring(705, 2);
		if (trackCode != "00")
			command->Parameters->AddWithValue("@track_code", trackCode);
		else
			command->Parameters->AddWithValue("@track_code", DBNull::Value);

		String^ courseKubun = record->Substring(709, 2);
		if (courseKubun != "  ")
			command->Parameters->AddWithValue("@course_kubun", courseKubun);
		else
			command->Parameters->AddWithValue("@course_kubun", DBNull::Value);

		String^ tenkoCode = record->Substring(887, 1);
		if (tenkoCode != "0")
			command->Parameters->AddWithValue("@tenko_code", tenkoCode);
		else
			command->Parameters->AddWithValue("@tenko_code", DBNull::Value);

		Int16 trackCodeInt = Int16::Parse(trackCode);
		String^ babajotaiCode = "0";
		if ((10 <= trackCodeInt && trackCodeInt <= 22) || trackCodeInt == 51 || (53 <= trackCodeInt && trackCodeInt <= 59))
			babajotaiCode = record->Substring(888, 1);
		else if ((23 <= trackCodeInt && trackCodeInt <= 29) || trackCodeInt == 52)
			babajotaiCode = record->Substring(889, 1);
		if (babajotaiCode != "0")
			command->Parameters->AddWithValue("@babajotai_code", babajotaiCode);
		else
			command->Parameters->AddWithValue("@babajotai_code", DBNull::Value);

		// 既存のレコードがあるか確認
		command->CommandText = "SELECT creation_date FROM ra WHERE "
			"kaisai_date = @kaisai_date AND "
			"keibajo_code = @keibajo_code AND "
			"kyoso_bango = @kyoso_bango";
		Object^ existingCreationDateObj = command->ExecuteScalar();

		// 既存のレコードがない場合は挿入して終了
		if (existingCreationDateObj == nullptr) {
			command->CommandText = "INSERT INTO ra (data_type, creation_date, "
				"kaisai_date, keibajo_code, kyoso_bango, "
				"kyori, track_code, course_kubun, tenko_code, babajotai_code) "
				"VALUES (@data_type, @creation_date, "
				"@kaisai_date, @keibajo_code, @kyoso_bango, "
				"@kyori, @track_code, @course_kubun, @tenko_code, @babajotai_code)";
			command->ExecuteNonQuery();
			return PROCESS_SUCCESS;
		}

		// 既存のレコードがある場合は日付を確認
		DateTime^ existingCreationDate = Convert::ToDateTime(existingCreationDateObj);

		// 既存の日付の方が新しい場合は何もしないで終了
		if (existingCreationDate->CompareTo(creationDate) > 0)
			return PROCESS_SUCCESS;

		// 既存の日付が古いか同じ場合は更新して終了
		command->CommandText = "UPDATE ra SET "
			"data_type = @data_type, "
			"creation_date = @creation_date, "
			"kyori = @kyori, "
			"track_code = @track_code, "
			"course_kubun = @course_kubun, "
			"tenko_code = @tenko_code, "
			"babajotai_code = @babajotai_code "
			"WHERE kaisai_date = @kaisai_date "
			"AND keibajo_code = @keibajo_code "
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
