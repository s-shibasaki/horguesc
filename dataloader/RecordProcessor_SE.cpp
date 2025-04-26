#include "RecordProcessor.h"

using namespace System;
using namespace Npgsql;


int RecordProcessor::ProcessSERecord(String^ record) {
	try {
		NpgsqlCommand^ command = gcnew NpgsqlCommand(nullptr, connection);

		command->Parameters->AddWithValue("@data_type", record->Substring(2, 1));

		DateTime^ creationDate = DateTime::ParseExact(record->Substring(3, 8), "yyyyMMdd", nullptr);
		command->Parameters->AddWithValue("@creation_date", creationDate);

		command->Parameters->AddWithValue("@kaisai_date", DateTime::ParseExact(record->Substring(11, 8), "yyyyMMdd", nullptr));
		command->Parameters->AddWithValue("@keibajo_code", record->Substring(19, 2));
		command->Parameters->AddWithValue("@kaisai_kai", Int16::Parse(record->Substring(21, 2)));
		command->Parameters->AddWithValue("@kaisai_nichime", Int16::Parse(record->Substring(23, 2)));
		command->Parameters->AddWithValue("@kyoso_bango", Int16::Parse(record->Substring(25, 2)));
		command->Parameters->AddWithValue("@wakuban", Int16::Parse(record->Substring(27, 1)));
		command->Parameters->AddWithValue("@umaban", Int16::Parse(record->Substring(28, 2)));
		command->Parameters->AddWithValue("@ketto_toroku_bango", Int64::Parse(record->Substring(30, 10)));
		command->Parameters->AddWithValue("@futan_juryo", Int16::Parse(record->Substring(288, 2)));
		command->Parameters->AddWithValue("@blinker_shiyo_kubun", record->Substring(294, 1));
		command->Parameters->AddWithValue("@kishu_code", Int32::Parse(record->Substring(296, 5)));
		command->Parameters->AddWithValue("@kishu_minarai_code", record->Substring(322, 1));

		command->Parameters->AddWithValue("@bataiju", DBNull::Value);
		String^ bataijuStr = record->Substring(324, 3);
		if (bataijuStr != "   ")
			command->Parameters->AddWithValue("@bataiju", Int16::Parse(bataijuStr));

		command->Parameters->AddWithValue("@zogensa", DBNull::Value);
		String^ zogensaStr = record->Substring(327, 4);
		if (zogensaStr != "    " && zogensaStr != " 999")
			command->Parameters->AddWithValue("@zogensa", Int16::Parse(zogensaStr));

		command->Parameters->AddWithValue("@ijo_kubun_code", record->Substring(331, 1));
		command->Parameters->AddWithValue("@kakutei_chakujun", Int16::Parse(record->Substring(334, 2)));
		command->Parameters->AddWithValue("@soha_time", Int16::Parse(record->Substring(338, 1)) * 600 + Int16::Parse(record->Substring(339, 3)));

		// 既存のレコードがあるか確認
		command->CommandText =
			"SELECT creation_date FROM se"
			" WHERE kaisai_date = @kaisai_date"
			" AND keibajo_code = @keibajo_code"
			" AND kaisai_kai = @kaisai_kai"
			" AND kaisai_nichime = @kaisai_nichime"
			" AND kyoso_bango = @kyoso_bango"
			" AND umaban = @umaban"
			" AND ketto_toroku_bango = @ketto_toroku_bango";
		Object^ existingCreationDateObj = command->ExecuteScalar();

		// 既存のレコードがない場合は挿入して終了
		if (existingCreationDateObj == nullptr) {
			command->CommandText =
				"INSERT INTO se (data_type, creation_date, "
				"kaisai_date, keibajo_code, kaisai_kai, kaisai_nichime, kyoso_bango, "
				"wakuban, umaban, ketto_toroku_bango, "
				"futan_juryo, blinker_shiyo_kubun, kishu_code, kishu_minarai_code, bataiju, zogensa, "
				"ijo_kubun_code, kakutei_chakujun, soha_time"
				") VALUES (@data_type, @creation_date, "
				"@kaisai_date, @keibajo_code, @kaisai_kai, @kaisai_nichime, @kyoso_bango, "
				"@wakuban, @umaban, @ketto_toroku_bango, "
				"@futan_juryo, @blinker_shiyo_kubun, @kishu_code, @kishu_minarai_code, @bataiju, @zogensa, "
				"@ijo_kubun_code, @kakutei_chakujun, @soha_time"
				")";
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
			"UPDATE se SET "
			"data_type = @data_type, "
			"creation_date = @creation_date, "
			"wakuban = @wakuban, "
			"futan_juryo = @futan_juryo, "
			"blinker_shiyo_kubun = @blinker_shiyo_kubun, "
			"kishu_code = @kishu_code, "
			"kishu_minarai_code = @kishu_minarai_code, "
			"bataiju = @bataiju, "
			"zogensa = @zogensa, "
			"ijo_kubun_code = @ijo_kubun_code, "
			"kakutei_chakujun = @kakutei_chakujun, "
			"soha_time = @soha_time "
			"WHERE kaisai_date = @kaisai_date "
			"AND keibajo_code = @keibajo_code "
			"AND kaisai_kai = @kaisai_kai "
			"AND kaisai_nichime = @kaisai_nichime "
			"AND kyoso_bango = @kyoso_bango "
			"AND umaban = @umaban "
			"AND ketto_toroku_bango = @ketto_toroku_bango";
		command->ExecuteNonQuery();
		return PROCESS_SUCCESS;
	}
	catch (Exception^ ex) {
		Console::WriteLine();
		Console::WriteLine("Error processing SE record: {0}", ex->Message);
		return PROCESS_ERROR;
	}
}