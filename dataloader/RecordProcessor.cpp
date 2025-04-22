#include "RecordProcessor.h"

using namespace System;
using namespace Npgsql;

RecordProcessor::RecordProcessor(NpgsqlConnection^ connection) {
	this->connection = connection;
}

bool RecordProcessor::Initialize() {
	try {
		NpgsqlCommand^ command = gcnew NpgsqlCommand(nullptr, connection);

		command->CommandText = "CREATE TABLE IF NOT EXISTS ra ("
			"id SERIAL PRIMARY KEY, "
			"creation_date DATE, "
			"data_type VARCHAR(1), "
			"kaisai_date DATE, "
			"keibajo_code VARCHAR(2), "
			"kaisai_kai INT, "
			"kaisai_nichime INT, "
			"kyoso_bango INT)";
		command->ExecuteNonQuery();

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
		// process ra record
		return PROCESS_SUCCESS;
	}

	else if (recordTypeId == "SE") {
		// process se record
		return PROCESS_SUCCESS;
	}

	else if (recordTypeId == "UM") {
		// process um record
		return PROCESS_SUCCESS;
	}

	return PROCESS_SKIP;
}

