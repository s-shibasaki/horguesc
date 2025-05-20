#include "RecordProcessor.h"

using namespace System;
using namespace Npgsql;

RecordProcessor::RecordProcessor(NpgsqlConnection^ connection) {
	this->connection = connection;
}

bool RecordProcessor::Initialize() {
	try {
		NpgsqlCommand^ command = gcnew NpgsqlCommand(nullptr, connection);

		command->CommandText =
			"CREATE TABLE IF NOT EXISTS ra ("
			"data_type CHAR(1), "
			"creation_date DATE, "
			"kaisai_date DATE, "
			"keibajo_code CHAR(2), "
			"kaisai_kai SMALLINT, "
			"kaisai_nichime SMALLINT, "
			"kyoso_bango SMALLINT, "
			"kyori SMALLINT, "
			"track_code CHAR(2), "
			"course_kubun CHAR(2), "
			"tenko_code CHAR(1), "
			"babajotai_code CHAR(1), "
			"hasso_time TIME, "
			"PRIMARY KEY (kaisai_date, keibajo_code, kaisai_kai, kaisai_nichime, kyoso_bango))";
		command->ExecuteNonQuery();

		command->CommandText =
			"CREATE TABLE IF NOT EXISTS se ("
			"data_type CHAR(1), "
			"creation_date DATE, "
			"kaisai_date DATE, "
			"keibajo_code CHAR(2), "
			"kaisai_kai SMALLINT, "
			"kaisai_nichime SMALLINT, "
			"kyoso_bango SMALLINT, "
			"wakuban SMALLINT, "
			"umaban SMALLINT, "
			"ketto_toroku_bango BIGINT, "
			"futan_juryo SMALLINT, "
			"blinker_shiyo_kubun CHAR(1), "
			"kishu_code INT, "
			"kishu_minarai_code CHAR(1), "
			"bataiju SMALLINT, "
			"zogensa SMALLINT, "
			"ijo_kubun_code CHAR(1), "
			"kakutei_chakujun SMALLINT, "
			"soha_time SMALLINT, "
			"PRIMARY KEY (kaisai_date, keibajo_code, kaisai_kai, kaisai_nichime, kyoso_bango, umaban, ketto_toroku_bango))";
		command->ExecuteNonQuery();

		command->CommandText =
			"CREATE TABLE IF NOT EXISTS um ("
			"data_type CHAR(1), "
			"creation_date DATE, "
			"ketto_toroku_bango BIGINT, "
			"birth_date DATE, "
			"seibetsu_code CHAR(1), ";
		for (int i = 1; i <= 14; i++) 
			command->CommandText += String::Format("hanshoku_toroku_bango_{0:00} BIGINT, ", i);
		command->CommandText += 
			"chokyoshi_code INT, "
			"PRIMARY KEY (ketto_toroku_bango))";
		command->ExecuteNonQuery();

		command->CommandText =
			"CREATE TABLE IF NOT EXISTS hn ("
			"data_type CHAR(1), "
			"creation_date DATE, "
			"hanshoku_toroku_bango BIGINT, "
			"ketto_toroku_bango BIGINT, "
			"PRIMARY KEY (hanshoku_toroku_bango))";
		command->ExecuteNonQuery();

		command->CommandText =
			"CREATE TABLE IF NOT EXISTS ch ("
			"data_type CHAR(1), "
			"creation_date DATE, "
			"chokyoshi_code INT, "
			"seibetsu_kubun CHAR(1), "
			"PRIMARY KEY (chokyoshi_code))";
		command->ExecuteNonQuery();

		command->CommandText =
			"CREATE TABLE IF NOT EXISTS ks ("
			"data_type CHAR(1), "
			"creation_date DATE, "
			"kishu_code INT, "
			"seibetsu_kubun CHAR(1), "
			"birth_date DATE, "  // Added birth_date field
			"PRIMARY KEY (kishu_code))";
		command->ExecuteNonQuery();

		command->CommandText =
			"CREATE TABLE IF NOT EXISTS cs ("
			"data_type CHAR(1), "
			"creation_date DATE, "
			"keibajo_code CHAR(2), "
			"kyori SMALLINT, "
			"track_code CHAR(2), "
			"kaishu_date DATE, "
			"description VARCHAR(6800), "
			"PRIMARY KEY (keibajo_code, kyori, track_code, kaishu_date))";
		command->ExecuteNonQuery();

		command->CommandText = 
			"CREATE TABLE IF NOT EXISTS we ("
			"data_type CHAR(1), "
			"creation_date DATE, "
			"kaisai_date DATE, "
			"keibajo_code CHAR(2), "
			"kaisai_kai SMALLINT, "
			"kaisai_nichime SMALLINT, "
			"happyo_datetime TIMESTAMP, "
			"henko_shikibetsu CHAR(1), "
			"tenko_code CHAR(1), "
			"babajotai_code_shiba CHAR(1), "
			"babajotai_code_dirt CHAR(1), "
			"PRIMARY KEY (kaisai_date, keibajo_code, kaisai_kai, kaisai_nichime, happyo_datetime, henko_shikibetsu))";
		command->ExecuteNonQuery();

		command->CommandText = 
			"CREATE TABLE IF NOT EXISTS av ("
			"data_type CHAR(1), "
			"creation_date DATE, "
			"kaisai_date DATE, "
			"keibajo_code CHAR(2), "
			"kaisai_kai SMALLINT, "
			"kaisai_nichime SMALLINT, "
			"kyoso_bango SMALLINT, "
			"umaban SMALLINT, "
			"PRIMARY KEY (kaisai_date, keibajo_code, kaisai_kai, kaisai_nichime, kyoso_bango, umaban))";
		command->ExecuteNonQuery();

		command->CommandText = 
			"CREATE TABLE IF NOT EXISTS jc ("
			"data_type CHAR(1), "
			"creation_date DATE, "
			"kaisai_date DATE, "
			"keibajo_code CHAR(2), "
			"kaisai_kai SMALLINT, "
			"kaisai_nichime SMALLINT, "
			"kyoso_bango SMALLINT, "
			"happyo_datetime TIMESTAMP, "
			"umaban SMALLINT, "
			"futan_juryo SMALLINT, "
			"kishu_code INT, "
			"kishu_minarai_code CHAR(1), "
			"PRIMARY KEY (kaisai_date, keibajo_code, kaisai_kai, kaisai_nichime, kyoso_bango, happyo_datetime, umaban))";
		command->ExecuteNonQuery();

		command->CommandText =
			"CREATE TABLE IF NOT EXISTS tc ("
			"data_type CHAR(1), "
			"creation_date DATE, "
			"kaisai_date DATE, "
			"keibajo_code CHAR(2), "
			"kaisai_kai SMALLINT, "
			"kaisai_nichime SMALLINT, "
			"kyoso_bango SMALLINT, "
			"hasso_time TIME, "
			"PRIMARY KEY (kaisai_date, keibajo_code, kaisai_kai, kaisai_nichime, kyoso_bango))";
		command->ExecuteNonQuery();

		command->CommandText =
			"CREATE TABLE IF NOT EXISTS cc ("
			"data_type CHAR(1), "
			"creation_date DATE, "
			"kaisai_date DATE, "
			"keibajo_code CHAR(2), "
			"kaisai_kai SMALLINT, "
			"kaisai_nichime SMALLINT, "
			"kyoso_bango SMALLINT, "
			"kyori SMALLINT, "
			"track_code CHAR(2), "
			"PRIMARY KEY (kaisai_date, keibajo_code, kaisai_kai, kaisai_nichime, kyoso_bango))";
		command->ExecuteNonQuery(); // この行を追加

		command->CommandText =
			"CREATE TABLE IF NOT EXISTS wh ("
			"data_type CHAR(1), "
			"creation_date DATE, "
			"kaisai_date DATE, "
			"keibajo_code CHAR(2), "
			"kaisai_kai SMALLINT, "
			"kaisai_nichime SMALLINT, "
			"kyoso_bango SMALLINT, "
			"umaban SMALLINT, "
			"bataiju SMALLINT, "
			"zogensa SMALLINT, "
			"PRIMARY KEY (kaisai_date, keibajo_code, kaisai_kai, kaisai_nichime, kyoso_bango, umaban))";
		command->ExecuteNonQuery();

		command->CommandText =
			"CREATE TABLE IF NOT EXISTS tansho ("
			"data_type CHAR(1), "
			"creation_date DATE, "
			"kaisai_date DATE, "
			"keibajo_code CHAR(2), "
			"kaisai_kai SMALLINT, "
			"kaisai_nichime SMALLINT, "
			"kyoso_bango SMALLINT, "
			"happyo_datetime TIMESTAMP, "
			"umaban SMALLINT, "
			"odds INT, "
			"PRIMARY KEY (kaisai_date, keibajo_code, kaisai_kai, kaisai_nichime, kyoso_bango, happyo_datetime, umaban))";
		command->ExecuteNonQuery();

		command->CommandText =
			"CREATE TABLE IF NOT EXISTS fukusho ("
			"data_type CHAR(1), "
			"creation_date DATE, "
			"kaisai_date DATE, "
			"keibajo_code CHAR(2), "
			"kaisai_kai SMALLINT, "
			"kaisai_nichime SMALLINT, "
			"kyoso_bango SMALLINT, "
			"happyo_datetime TIMESTAMP, "
			"umaban SMALLINT, "
			"min_odds INT, "
			"max_odds INT, "
			"PRIMARY KEY (kaisai_date, keibajo_code, kaisai_kai, kaisai_nichime, kyoso_bango, happyo_datetime, umaban))";
		command->ExecuteNonQuery();

		command->CommandText =
			"CREATE TABLE IF NOT EXISTS wakuren ("
			"data_type CHAR(1), "
			"creation_date DATE, "
			"kaisai_date DATE, "
			"keibajo_code CHAR(2), "
			"kaisai_kai SMALLINT, "
			"kaisai_nichime SMALLINT, "
			"kyoso_bango SMALLINT, "
			"happyo_datetime TIMESTAMP, "
			"wakuban_1 SMALLINT, "
			"wakuban_2 SMALLINT, "
			"odds INT, "
			"PRIMARY KEY (kaisai_date, keibajo_code, kaisai_kai, kaisai_nichime, kyoso_bango, happyo_datetime, wakuban_1, wakuban_2))";
		command->ExecuteNonQuery();

		command->CommandText =
			"CREATE TABLE IF NOT EXISTS umaren ("
			"data_type CHAR(1), "
			"creation_date DATE, "
			"kaisai_date DATE, "
			"keibajo_code CHAR(2), "
			"kaisai_kai SMALLINT, "
			"kaisai_nichime SMALLINT, "
			"kyoso_bango SMALLINT, "
			"happyo_datetime TIMESTAMP, "
			"umaban_1 SMALLINT, "
			"umaban_2 SMALLINT, "
			"odds INT, "
			"PRIMARY KEY (kaisai_date, keibajo_code, kaisai_kai, kaisai_nichime, kyoso_bango, happyo_datetime, umaban_1, umaban_2))";
		command->ExecuteNonQuery();

		command->CommandText =
			"CREATE TABLE IF NOT EXISTS wide ("
			"data_type CHAR(1), "
			"creation_date DATE, "
			"kaisai_date DATE, "
			"keibajo_code CHAR(2), "
			"kaisai_kai SMALLINT, "
			"kaisai_nichime SMALLINT, "
			"kyoso_bango SMALLINT, "
			"happyo_datetime TIMESTAMP, "
			"umaban_1 SMALLINT, "
			"umaban_2 SMALLINT, "
			"min_odds INT, "
			"max_odds INT, "
			"PRIMARY KEY (kaisai_date, keibajo_code, kaisai_kai, kaisai_nichime, kyoso_bango, happyo_datetime, umaban_1, umaban_2))";
		command->ExecuteNonQuery();

		command->CommandText =
			"CREATE TABLE IF NOT EXISTS umatan ("
			"data_type CHAR(1), "
			"creation_date DATE, "
			"kaisai_date DATE, "
			"keibajo_code CHAR(2), "
			"kaisai_kai SMALLINT, "
			"kaisai_nichime SMALLINT, "
			"kyoso_bango SMALLINT, "
			"happyo_datetime TIMESTAMP, "
			"umaban_1 SMALLINT, "
			"umaban_2 SMALLINT, "
			"odds INT, "
			"PRIMARY KEY (kaisai_date, keibajo_code, kaisai_kai, kaisai_nichime, kyoso_bango, happyo_datetime, umaban_1, umaban_2))";
		command->ExecuteNonQuery();

		command->CommandText =
			"CREATE TABLE IF NOT EXISTS sanrenpuku ("
			"data_type CHAR(1), "
			"creation_date DATE, "
			"kaisai_date DATE, "
			"keibajo_code CHAR(2), "
			"kaisai_kai SMALLINT, "
			"kaisai_nichime SMALLINT, "
			"kyoso_bango SMALLINT, "
			"happyo_datetime TIMESTAMP, "
			"umaban_1 SMALLINT, "
			"umaban_2 SMALLINT, "
			"umaban_3 SMALLINT, "
			"odds INT, "
			"PRIMARY KEY (kaisai_date, keibajo_code, kaisai_kai, kaisai_nichime, kyoso_bango, happyo_datetime, umaban_1, umaban_2, umaban_3))";
		command->ExecuteNonQuery();

		command->CommandText =
			"CREATE TABLE IF NOT EXISTS sanrentan ("
			"data_type CHAR(1), "
			"creation_date DATE, "
			"kaisai_date DATE, "
			"keibajo_code CHAR(2), "
			"kaisai_kai SMALLINT, "
			"kaisai_nichime SMALLINT, "
			"kyoso_bango SMALLINT, "
			"happyo_datetime TIMESTAMP, "
			"umaban_1 SMALLINT, "
			"umaban_2 SMALLINT, "
			"umaban_3 SMALLINT, "
			"odds INT, "
			"PRIMARY KEY (kaisai_date, keibajo_code, kaisai_kai, kaisai_nichime, kyoso_bango, happyo_datetime, umaban_1, umaban_2, umaban_3))";
		command->ExecuteNonQuery();

		return true;
	}
	catch (Exception^ ex) {
		Console::WriteLine("Failed to create tables: {0}", ex->Message);
		return false;
	}
}

// PROCESS_ERROR を返す場合は改行すること
int RecordProcessor::ProcessRecord(array<Byte>^ record) {
	try {
		String^ recordTypeId = ByteSubstring(record, 0, 2);
		if (recordTypeId == "RA")
			return ProcessRARecord(record);
		else if (recordTypeId == "SE")
			return ProcessSERecord(record);
		else if (recordTypeId == "UM")
			return ProcessUMRecord(record);
		else if (recordTypeId == "HN")
			return ProcessHNRecord(record);
		else if (recordTypeId == "CH")
			return ProcessCHRecord(record);
		else if (recordTypeId == "KS")
			return ProcessKSRecord(record);
		else if (recordTypeId == "CS")
			return ProcessCSRecord(record);
		else if (recordTypeId == "WE")
			return ProcessWERecord(record);
		else if (recordTypeId == "AV")
			return ProcessAVRecord(record);
		else if (recordTypeId == "JC")
			return ProcessJCRecord(record);
		else if (recordTypeId == "TC")
			return ProcessTCRecord(record);
		else if (recordTypeId == "CC")
			return ProcessCCRecord(record);
		else if (recordTypeId == "WH")
			return ProcessWHRecord(record);
		else if (recordTypeId == "O1")
			return ProcessO1Record(record);
		else if (recordTypeId == "O2")
			return ProcessO2Record(record);
		else if (recordTypeId == "O3")
			return ProcessO3Record(record);
		else if (recordTypeId == "O4")
			return ProcessO4Record(record);
		else if (recordTypeId == "O5")
			return ProcessO5Record(record);
		else if (recordTypeId == "O6")
			return ProcessO6Record(record);
		else
			return PROCESS_SKIP; // Unknown record type, skip it
	}
	catch (Exception^ ex) {
		Console::WriteLine();
		Console::WriteLine("Error processing record: {0}", ex->Message);
		return PROCESS_ERROR;
	}
}

String^ RecordProcessor::ByteSubstring(array<Byte>^ bytes, int byteStartIndex, int byteLength) {
	// Ensure we don't exceed array bounds
	if (byteStartIndex >= bytes->Length)
		return String::Empty;

	// Adjust byteLenght if it would exceed the array
	if (byteStartIndex + byteLength > bytes->Length)
		byteLength = bytes->Length - byteStartIndex;

	// Extract the specified bytes and convert back to string
	return System::Text::Encoding::GetEncoding(932)->GetString(bytes, byteStartIndex, byteLength);
}