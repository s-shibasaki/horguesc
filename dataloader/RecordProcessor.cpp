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
			"PRIMARY KEY (kaisai_date, keibajo_code, kaisai_kai, kaisai_nichime, kyoso_bango))";
		command->ExecuteNonQuery();

		command->CommandText =
			"CREATE TABLE IF NOT EXISTS se ("
			"data_type VARCHAR(1), "
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
			"data_type VARCHAR(1), "
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
			"data_type VARCHAR(1), "
			"creation_date DATE, "
			"hanshoku_toroku_bango BIGINT, "
			"ketto_toroku_bango BIGINT, "
			"PRIMARY KEY (hanshoku_toroku_bango))";
		command->ExecuteNonQuery();

		command->CommandText =
			"CREATE TABLE IF NOT EXISTS ch ("
			"data_type VARCHAR(1), "
			"creation_date DATE, "
			"chokyoshi_code INT, "
			"seibetsu_kubun CHAR(1), "
			"PRIMARY KEY (chokyoshi_code))";
		command->ExecuteNonQuery();

		command->CommandText =
			"CREATE TABLE IF NOT EXISTS ks ("
			"data_type VARCHAR(1), "
			"creation_date DATE, "
			"kishu_code INT, "
			"seibetsu_kubun CHAR(1), "
			"birth_date DATE, "  // Added birth_date field
			"PRIMARY KEY (kishu_code))";
		command->ExecuteNonQuery();

		command->CommandText =
			"CREATE TABLE IF NOT EXISTS cs ("
			"data_type VARCHAR(1), "
			"creation_date DATE, "
			"keibajo_code CHAR(2), "
			"kyori SMALLINT, "
			"track_code CHAR(2), "
			"kaishu_date DATE, "
			"description VARCHAR(6800), "
			"PRIMARY KEY (keibajo_code, kyori, track_code, kaishu_date))";
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