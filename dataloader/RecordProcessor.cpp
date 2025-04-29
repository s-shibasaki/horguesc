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
			"PRIMARY KEY (kaisai_date, keibajo_code, kyoso_bango, umaban, ketto_toroku_bango))";
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



		return true;
	}
	catch (Exception^ ex) {
		Console::WriteLine("Failed to create tables: {0}", ex->Message);
		return false;
	}
}

// PROCESS_ERROR ‚ð•Ô‚·ê‡‚Í‰üs‚·‚é‚±‚Æ
int RecordProcessor::ProcessRecord(String^ record) {
	try {
		String^ recordTypeId = record->Substring(0, 2);
		int result = PROCESS_SKIP;

		if (recordTypeId == "RA")
			result = ProcessRARecord(record);
		else if (recordTypeId == "SE")
			result = ProcessSERecord(record);
		else if (recordTypeId == "UM")
			result = ProcessUMRecord(record);
		else if (recordTypeId == "HN")
			result = ProcessHNRecord(record);

		return result;
	}
	catch (Exception^ ex) {
		Console::WriteLine();
		Console::WriteLine("Error processing record: {0}", ex->Message);
		return PROCESS_ERROR;
	}
}

