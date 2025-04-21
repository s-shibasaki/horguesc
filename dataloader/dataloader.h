#pragma once

namespace dataloader {
	ref class DataLoader
	{
	private:
		AxJVDTLabLib::AxJVLink^ jvlink;

	public:
		DataLoader(AxJVDTLabLib::AxJVLink^ jvlink);

		bool Execute();
	};
}

