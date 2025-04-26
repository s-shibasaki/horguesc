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
		command->Parameters->AddWithValue("@kyoso_bango", Int16::Parse(record->Substring(25, 2)));

		Int16 wakuban = Int16::Parse(record->Substring(27, 1));
		if (wakuban > 0)
			command->Parameters->AddWithValue("@wakuban", wakuban);
		else
			command->Parameters->AddWithValue("@wakuban", DBNull::Value);

		command->Parameters->AddWithValue("@umaban", Int16::Parse(record->Substring(28, 2)));

		Int64 kettoTorokuBango = Int64::Parse(record->Substring(30, 10));
		if (kettoTorokuBango > 0)
			command->Parameters->AddWithValue("@ketto_toroku_bango", kettoTorokuBango);
		else
			command->Parameters->AddWithValue("@ketto_toroku_bango", DBNull::Value);

		Int16 futanJuryo = Int16::Parse(record->Substring(288, 2));
		if (futanJuryo > 0)
			command->Parameters->AddWithValue("@futan_juryo", futanJuryo);
		else
			command->Parameters->AddWithValue("@futan_juryo", DBNull::Value);

		String^ blinkerShiyoKubun = record->Substring(294, 1);
		if (blinkerShiyoKubun != "0")
			command->Parameters->AddWithValue("@blinker_shiyo_kubun", blinkerShiyoKubun);
		else
			command->Parameters->AddWithValue("@blinker_shiyo_kubun", DBNull::Value);

		Int32 kishuCode = Int32::Parse(record->Substring(296, 5));
		if (kishuCode > 0)
			command->Parameters->AddWithValue("@kishu_code", kishuCode);
		else
			command->Parameters->AddWithValue("@kishu_code", DBNull::Value);

		String^ kishuMinaraiCode = record->Substring(322, 1);
		if (kishuMinaraiCode != "0")
			command->Parameters->AddWithValue("@kishu_minarai_code", kishuMinaraiCode);
		else
			command->Parameters->AddWithValue("@kishu_minarai_code", DBNull::Value);

		String^ bataijuStr = record->Substring(324, 3);
		Int16 bataiju = 0;
		if (bataijuStr != "   ")
			bataiju = Int16::Parse(bataijuStr);
		if (2 <= bataiju && bataiju <= 998)
			command->Parameters->AddWithValue("@bataiju", bataiju);
		else
			command->Parameters->AddWithValue("@bataiju", DBNull::Value);

		String^ zogensaStr = record->Substring(327, 4);
		if (zogensaStr != "    " && zogensaStr != " 999")
			command->Parameters->AddWithValue("@zogensa", Int16::Parse(zogensaStr));
		else
			command->Parameters->AddWithValue("@zogensa", DBNull::Value);

		String^ ijoKubunCode = record->Substring(331, 1);
		if (ijoKubunCode != "0")
			command->Parameters->AddWithValue("@ijo_kubun_code", ijoKubunCode);
		else
			command->Parameters->AddWithValue("@ijo_kubun_code", DBNull::Value);

		Int16 kakuteiChakujun = Int16::Parse(record->Substring(334, 2));
		if (kakuteiChakujun > 0)
			command->Parameters->AddWithValue("@kakutei_chakujun", kakuteiChakujun);
		else
			command->Parameters->AddWithValue("@kakutei_chakujun", DBNull::Value);

		Int16 sohaTime = Int16::Parse(record->Substring(338, 1)) * 600 + Int16::Parse(record->Substring(339, 3));
		if (sohaTime > 0)
			command->Parameters->AddWithValue("@soha_time", sohaTime);
		else
			command->Parameters->AddWithValue("@soha_time", DBNull::Value);

		// 既存のレコードがあるか確認
		command->CommandText =
			"SELECT creation_date FROM se"
			" WHERE kaisai_date = @kaisai_date"
			" AND keibajo_code = @keibajo_code"
			" AND kyoso_bango = @kyoso_bango"
			" AND umaban = @umaban"
			" AND ketto_toroku_bango = @ketto_toroku_bango";
		Object^ existingCreationDateObj = command->ExecuteScalar();

		// 既存のレコードがない場合は挿入して終了
		if (existingCreationDateObj == nullptr) {
			command->CommandText =
				"INSERT INTO se (data_type, creation_date, "
				"kaisai_date, keibajo_code, kyoso_bango, "
				"wakuban, umaban, ketto_toroku_bango, "
				"futan_juryo, blinker_shiyo_kubun, kishu_code, kishu_minarai_code, bataiju, zogensa, "
				"ijo_kubun_code, kakutei_chakujun, soha_time"
				") VALUES (@data_type, @creation_date, "
				"@kaisai_date, @keibajo_code, @kyoso_bango, "
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