#include "RecordProcessor.h"

using namespace System;
using namespace Npgsql;

RecordProcessor::RecordProcessor(NpgsqlConnection^ connection) {
	this->connection = connection;
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

