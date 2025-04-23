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
			"creation_date DATE, "
			"data_type VARCHAR(1), "
			"kaisai_date DATE, "
			"keibajo_code VARCHAR(2), "
			"kaisai_kai INT, "
			"kaisai_nichime INT, "
			"kyoso_bango INT, "
			"kyori INT, "
			"track_code VARCHAR(2), "
			"course_kubun VARCHAR(2), "
			"tenko_code VARCHAR(1), "
			"babajotai_code VARCHAR(1))";
		command->ExecuteNonQuery();

		command->CommandText =
			"CREATE TABLE IF NOT EXISTS se ("
			"id SERIAL PRIMARY KEY, "
			"creation_date DATE, "
			"data_type VARCHAR(1), "
			"ra_id INT, "
			"wakuban INT, "
			"umaban INT, "
			"um_id INT)";
		command->ExecuteNonQuery();

		command->CommandText =
			"CREATE INDEX IF NOT EXISTS idx_ra_key ON ra ("
			"kaisai_date, "
			"keibajo_code, "
			"kaisai_kai, "
			"kaisai_nichime, "
			"kyoso_bango)";
		command->ExecuteNonQuery();

		command->CommandText =
			"CREATE INDEX IF NOT EXISTS idx_se_key ON se ("
			"ra_id, "
			"umaban)";

		return true;
	}
	catch (Exception^ ex) {
		Console::WriteLine("Failed to create tables: {0}", ex->Message);
		return false;
	}
}

int RecordProcessor::ProcessRecord(String^ record) {
	String^ recordTypeId = record->Substring(0, 2);

	if (recordTypeId == "RA") {
		return PROCESS_SUCCESS;
	}

	else if (recordTypeId == "SE") {
		// process se record
		return PROCESS_SUCCESS;
	}

	else if (recordTypeId == "UM") {
		return PROCESS_SKIP;
	}

	return PROCESS_SKIP;
}

