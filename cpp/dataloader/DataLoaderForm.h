#pragma once

namespace dataloader {

	using namespace System;
	using namespace System::ComponentModel;
	using namespace System::Collections;
	using namespace System::Windows::Forms;
	using namespace System::Data;
	using namespace System::Drawing;
	using Collections::Generic::List;

	public ref class DataLoaderForm : public System::Windows::Forms::Form {
	public:
		DataLoaderForm(array<String^>^ args) {
			InitializeComponent();
			this->args = args;
		}

	protected:
		~DataLoaderForm() {
			if (components)
			{
				delete components;
			}
		}

	private:
		AxJVDTLabLib::AxJVLink^ jvlink;
		System::ComponentModel::Container^ components;
		array<String^>^ args;

		void InitializeComponent(void) {
			System::ComponentModel::ComponentResourceManager^ resources = (gcnew System::ComponentModel::ComponentResourceManager(DataLoaderForm::typeid));
			this->jvlink = (gcnew AxJVDTLabLib::AxJVLink());
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->jvlink))->BeginInit();
			this->SuspendLayout();

			this->jvlink->Enabled = true;
			this->jvlink->Location = System::Drawing::Point(13, 13);
			this->jvlink->Name = L"jvlink";
			this->jvlink->OcxState = (cli::safe_cast<System::Windows::Forms::AxHost::State^>(resources->GetObject(L"jvlink.OcxState")));
			this->jvlink->Size = System::Drawing::Size(192, 192);
			this->jvlink->TabIndex = 0;

			this->AutoScaleDimensions = System::Drawing::SizeF(6, 13);
			this->AutoScaleMode = System::Windows::Forms::AutoScaleMode::Font;
			this->ClientSize = System::Drawing::Size(218, 217);
			this->Controls->Add(this->jvlink);
			this->Name = L"DataLoaderForm";
			this->Text = L"Data Loader";
			this->Shown += gcnew System::EventHandler(this, &DataLoaderForm::form_Shown);
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->jvlink))->EndInit();
			this->ResumeLayout(false);
		}

		ref struct DataFetchParams {
			String^ dataSpec;
			String^ fromTime;
			int option;
			DataFetchParams(String^ dataSpec, String^ fromTime, int option) {
				this->dataSpec = dataSpec;
				this->fromTime = fromTime;
				this->option = option;
			}
		};

		bool FetchAndAppendData(DataFetchParams^ params, IO::StreamWriter^ writer) {
			try {
				// params
				int size = 110000;

				// variables for jvlink
				int retVal;
				int readCount;
				int downloadCount;
				String^ lastFileTimeStamp;
				String^ fileName;
				String^ buff;

				// other variables
				int progress;
				int lastRetVal;

				// show dataSpec, fromTime, option
				Console::WriteLine("--------------------------------------------------");
				Console::WriteLine("Data Spec: " + params->dataSpec);
				Console::WriteLine("From Time: " + params->fromTime);
				Console::WriteLine("Option: " + params->option);

				// JVOpen
				retVal = jvlink->JVOpen(params->dataSpec, params->fromTime, params->option, readCount, downloadCount, lastFileTimeStamp);
				if (retVal == 0) {
					Console::WriteLine("JVLink opened successfully.");
				}
				else {
					Console::WriteLine("JVLink open failed with error code: " + retVal.ToString());
					return false;
				}

				// show readCount, downloadCount, lastFileTimeStamp
				Console::WriteLine("Read Count: " + readCount.ToString());
				Console::WriteLine("Download Count: " + downloadCount.ToString());
				Console::WriteLine("Last File Time Stamp: " + lastFileTimeStamp);

				// JVStatus
				retVal = 0;
				lastRetVal = -1;
				while (true) {
					if (retVal >= 0) {
						if (lastRetVal != retVal) {
							lastRetVal = retVal;
							Console::Write("\rDownload progress: " + retVal.ToString() + " / " + downloadCount.ToString());

							if (retVal == downloadCount) {
								// all files downloaded
								Console::WriteLine("\r\nAll files downloaded.");
								break;
							}
						}
					}
					else {
						// download failed
						Console::WriteLine("\r\nDownload failed with error code: " + retVal.ToString());
						return false;
					}
					retVal = jvlink->JVStatus();
				}

				// JVRead
				progress = -1;
				retVal = -1;
				while (true) {
					if (retVal > 0) {
						writer->Write(buff);
					}
					else if (retVal == -1) {
						++progress;
						Console::Write("\rReading progress: " + progress.ToString() + " / " + readCount.ToString());
					}
					else if (retVal == 0) {
						// all files read
						Console::WriteLine("\r\nAll files read.");
						break;
					}
					else {
						// read failed
						Console::WriteLine("\r\nRead failed with error code: " + retVal.ToString());
						return false;
					}
					retVal = jvlink->JVRead(buff, size, fileName);
				}

				return true;
			}
			finally {
				jvlink->JVClose();
			}
		}

		System::Void form_Shown(System::Object^ sender, System::EventArgs^ e) {
			IO::StreamWriter^ writer = nullptr;
			bool success = false;
			try {
				// params
				String^ sid = "UNKNOWN";
				int fromYear = 2020;
				int currentYear = 2025;

				// valiables for jvlink
				int retVal;

				List<DataFetchParams^>^ paramsList = gcnew List<DataFetchParams^>();
				for (int year = fromYear; year < currentYear; year++) {
					paramsList->Add(gcnew DataFetchParams("RACEBLDNSNPNSLOPWOODYSCHMING", year.ToString() + "0101000000-" + (year + 1).ToString() + "0101000000", 4));
				}
				paramsList->Add(gcnew DataFetchParams("RACEBLDNSNPNSLOPWOODYSCHMINGTOKUDIFNHOSNHOYUCOMM", currentYear.ToString() + "0101000000", 4));

				// JVInit
				retVal = jvlink->JVInit(sid);
				if (retVal == 0) {
					Console::WriteLine("JVLink initialized successfully.");
				}
				else {
					Console::WriteLine("JVLink initialization failed with error code: " + retVal.ToString());
					return;
				}

				// Check if output directory exists, create if not
				String^ outputDir = "data";
				if (!IO::Directory::Exists(outputDir)) {
					IO::Directory::CreateDirectory(outputDir);
					Console::WriteLine("Created output directory: " + outputDir);
				}

				writer = gcnew IO::StreamWriter("data/jvd.dat", false);
				Console::WriteLine("Created file data/jvd.dat");

				for (int i = 0; i < paramsList->Count; i++) {
					Console::WriteLine("\nProcessing data set " + (i + 1).ToString() + " / " + paramsList->Count.ToString());

					bool success = FetchAndAppendData(paramsList[i], writer);
					if (!success) {
						Console::WriteLine("Failed to process data set " + (i + 1).ToString() + ". Aborting.");
						return;
					}
				}

				Console::WriteLine("\nAll data processing completed.");
				success = true;
				return;
			}
			finally {
				if (writer != nullptr) {
					writer->Close();
				}
				if (success) {
					Environment::Exit(0);
				}
				else {
					Environment::Exit(1);
				}
			}
		}
	};
}