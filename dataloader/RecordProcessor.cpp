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
			"id SERIAL PRIMARY KEY, "
			"data_type CHAR(1), "
			"creation_date DATE, "
			"kaisai_date DATE, "
			"keibajo_code CHAR(2), "
			"kyoso_bango SMALLINT, "
			"kyori SMALLINT, "
			"track_code CHAR(2), "
			"course_kubun CHAR(2), "
			"tenko_code CHAR(1), "
			"babajotai_code CHAR(1)); "
			"ALTER TABLE ra DROP CONSTRAINT IF EXISTS ra_unique_constraint; "
			"ALTER TABLE ra ADD CONSTRAINT ra_unique_constraint UNIQUE "
			"(kaisai_date, keibajo_code, kyoso_bango)";
		command->ExecuteNonQuery();

		command->CommandText =
			"CREATE TABLE IF NOT EXISTS se ("
			"id SERIAL PRIMARY KEY, "
			"data_type VARCHAR(1), "
			"creation_date DATE, "
			"kaisai_date DATE, "
			"keibajo_code CHAR(2), "
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
			"soha_time SMALLINT); "
			"ALTER TABLE se DROP CONSTRAINT IF EXISTS se_unique_constraint; "
			"ALTER TABLE se ADD CONSTRAINT se_unique_constraint UNIQUE "
			"(kaisai_date, keibajo_code, kyoso_bango, umaban, ketto_toroku_bango)";
		command->ExecuteNonQuery();

		command->CommandText =
			"CREATE TABLE IF NOT EXISTS um ("
			"id SERIAL PRIMARY KEY, "
			"data_type VARCHAR(1), "
			"creation_date DATE, "
			"ketto_toroku_bango BIGINT, "
			"birth_date DATE, "
			"seibetsu_code CHAR(1), "
			"hanshoku_toroku_bango_01 BIGINT, "
			"hanshoku_toroku_bango_02 BIGINT, "
			"hanshoku_toroku_bango_03 BIGINT, "
			"hanshoku_toroku_bango_04 BIGINT, "
			"hanshoku_toroku_bango_05 BIGINT, "
			"hanshoku_toroku_bango_06 BIGINT, "
			"hanshoku_toroku_bango_07 BIGINT, "
			"hanshoku_toroku_bango_08 BIGINT, "
			"hanshoku_toroku_bango_09 BIGINT, "
			"hanshoku_toroku_bango_10 BIGINT, "
			"hanshoku_toroku_bango_11 BIGINT, "
			"hanshoku_toroku_bango_12 BIGINT, "
			"hanshoku_toroku_bango_13 BIGINT, "
			"hanshoku_toroku_bango_14 BIGINT, "
			"chokyoshi_code INT); "
			"ALTER TABLE um DROP CONSTRAINT IF EXISTS um_unique_constraint; "
			"ALTER TABLE um ADD CONSTRAINT um_unique_constraint UNIQUE "
			"(ketto_toroku_bango)";
		command->ExecuteNonQuery();

		command->CommandText =
			"CREATE TABLE IF NOT EXISTS hn ("
			"id SERIAL PRIMARY KEY, "
			"data_type VARCHAR(1), "
			"creation_date DATE, "
			"hanshoku_toroku_bango BIGINT, "
			"ketto_toroku_bango BIGINT); "
			"ALTER TABLE hn DROP CONSTRAINT IF EXISTS hn_unique_constraint; "
			"ALTER TABLE hn ADD CONSTRAINT hn_unique_constraint UNIQUE "
			"(hanshoku_toroku_bango)";
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

