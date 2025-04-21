#include "dataloader.h"

namespace dataloader {
	
	DataLoader::DataLoader(AxJVDTLabLib::AxJVLink^ jvlink) : jvlink(jvlink) {}

	bool DataLoader::Execute() {
		return true;
	}

}