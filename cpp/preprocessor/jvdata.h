#pragma once

#include <string>
#include <array>
#include <sstream>

namespace JVData
{

	//////////////////// 共通クラス ////////////////////

	// 年月日
	class YMD
	{
	public:
		std::string Year;  // 年
		std::string Month; // 月
		std::string Day;   // 日
		YMD() : Year(""), Month(""), Day("") {}
		YMD(std::string ymd)
		{
			if (ymd.size() != 8)
			{
				throw std::invalid_argument("Invalid YMD format. Expected 8 characters.");
			}
			Year = ymd.substr(0, 4);
			Month = ymd.substr(4, 2);
			Day = ymd.substr(6, 2);
		}

		std::string toString() const
		{
			std::stringstream ss;
			ss << "YMD{Year=" << Year << ", Month=" << Month << ", Day=" << Day << "}";
			return ss.str();
		}
	};

	// 時分秒
	class HMS
	{
	public:
		std::string Hour;	// 時
		std::string Minute; // 分
		std::string Second; // 秒
		HMS() : Hour(""), Minute(""), Second("") {}
		HMS(std::string hms)
		{
			if (hms.size() != 6)
			{
				throw std::invalid_argument("Invalid HMS format. Expected 6 characters.");
			}
			Hour = hms.substr(0, 2);
			Minute = hms.substr(2, 2);
			Second = hms.substr(4, 2);
		}

		std::string toString() const
		{
			std::stringstream ss;
			ss << "HMS{Hour=" << Hour << ", Minute=" << Minute << ", Second=" << Second << "}";
			return ss.str();
		}
	};

	// 時分
	class HM
	{
	public:
		std::string Hour;	// 時
		std::string Minute; // 分
		HM() : Hour(""), Minute("") {}
		HM(std::string hm)
		{
			if (hm.size() != 4)
			{
				throw std::invalid_argument("Invalid HM format. Expected 4 characters.");
			}
			Hour = hm.substr(0, 2);
			Minute = hm.substr(2, 2);
		}

		std::string toString() const
		{
			std::stringstream ss;
			ss << "HM{Hour=" << Hour << ", Minute=" << Minute << "}";
			return ss.str();
		}
	};

	// 月日時分
	class MDHM
	{
	public:
		std::string Month;	// 月
		std::string Day;	// 日
		std::string Hour;	// 時
		std::string Minute; // 分
		MDHM() : Month(""), Day(""), Hour(""), Minute("") {}
		MDHM(std::string mdhm)
		{
			if (mdhm.size() != 8)
			{
				throw std::invalid_argument("Invalid MDHM format. Expected 8 characters.");
			}
			Month = mdhm.substr(0, 2);
			Day = mdhm.substr(2, 2);
			Hour = mdhm.substr(4, 2);
			Minute = mdhm.substr(6, 2);
		}

		std::string toString() const
		{
			std::stringstream ss;
			ss << "MDHM{Month=" << Month << ", Day=" << Day << ", Hour=" << Hour << ", Minute=" << Minute << "}";
			return ss.str();
		}
	};

	// レコードヘッダ
	class RECORD_ID
	{
	public:
		std::string RecordSpec; // レコード種別
		std::string DataKubun;	// データ区分
		YMD MakeDate;			// データ作成年月日
		RECORD_ID() : RecordSpec(""), DataKubun(""), MakeDate() {}
		RECORD_ID(std::string record_id)
		{
			if (record_id.size() != 11)
			{
				throw std::invalid_argument("Invalid RECORD_ID format. Expected 11 characters.");
			}
			RecordSpec = record_id.substr(0, 2);
			DataKubun = record_id.substr(2, 1);
			MakeDate = YMD(record_id.substr(3, 8));
		}

		std::string toString() const
		{
			std::stringstream ss;
			ss << "RECORD_ID{RecordSpec=" << RecordSpec << ", DataKubun=" << DataKubun << ", MakeDate=" << MakeDate.toString() << "}";
			return ss.str();
		}
	};

	// 競走識別情報
	class RACE_ID
	{
	public:
		std::string Year;	  // 開催年
		std::string MonthDay; // 開催月日
		std::string JyoCD;	  // 競馬場コード
		std::string Kaiji;	  // 開催回[第N回]
		std::string Nichiji;  // 開催日目[N日目]
		std::string RaceNum;  // レース番号
		RACE_ID() : Year(""), MonthDay(""), JyoCD(""), Kaiji(""), Nichiji(""), RaceNum("") {}
		RACE_ID(std::string race_id)
		{
			if (race_id.size() == 16)
			{
				Year = race_id.substr(0, 4);
				MonthDay = race_id.substr(4, 4);
				JyoCD = race_id.substr(8, 2);
				Kaiji = race_id.substr(10, 2);
				Nichiji = race_id.substr(12, 2);
				RaceNum = race_id.substr(14, 2);
			}
			else if (race_id.size() == 14)
			{
				Year = race_id.substr(0, 4);
				MonthDay = race_id.substr(4, 4);
				JyoCD = race_id.substr(8, 2);
				Kaiji = race_id.substr(10, 2);
				Nichiji = race_id.substr(12, 2);
				RaceNum = "00";
			}
			else
			{
				throw std::invalid_argument("Invalid RACE_ID format. Expected 14 or 16 characters.");
			}
		}

		std::string toString() const
		{
			std::stringstream ss;
			ss << "RACE_ID{Year=" << Year << ", MonthDay=" << MonthDay << ", JyoCD=" << JyoCD << ", Kaiji=" << Kaiji << ", Nichiji=" << Nichiji << ", RaceNum=" << RaceNum << "}";
			return ss.str();
		}
	};

	// 着回数
	class CHAKUKAISU_INFO
	{
	public:
		std::array<std::string, 6> Chakukaisu; // 着回数
		CHAKUKAISU_INFO()
		{
			for (int i = 0; i < 6; ++i)
			{
				Chakukaisu[i] = "";
			}
		}
		CHAKUKAISU_INFO(std::string chakukaisu_info)
		{
			if (chakukaisu_info.size() == 18)
			{
				for (int i = 0; i < 6; ++i)
				{
					Chakukaisu[i] = chakukaisu_info.substr(i * 3, 3);
				}
			}
			else if (chakukaisu_info.size() == 24)
			{
				for (int i = 0; i < 6; ++i)
				{
					Chakukaisu[i] = chakukaisu_info.substr(i * 4, 4);
				}
			}
			else if (chakukaisu_info.size() == 30)
			{
				for (int i = 0; i < 6; ++i)
				{
					Chakukaisu[i] = chakukaisu_info.substr(i * 5, 5);
				}
			}
			else if (chakukaisu_info.size() == 36)
			{
				for (int i = 0; i < 6; ++i)
				{
					Chakukaisu[i] = chakukaisu_info.substr(i * 6, 6);
				}
			}
			else
			{
				throw std::invalid_argument("Invalid CHAKUKAISU_INFO format. Expected 18, 24, 30, or 36 characters.");
			}
		}

		std::string toString() const
		{
			std::stringstream ss;
			ss << "CHAKUKAISU_INFO{";
			for (size_t i = 0; i < Chakukaisu.size(); ++i)
			{
				ss << "Chakukaisu[" << i << "]=" << Chakukaisu[i] << (i < Chakukaisu.size() - 1 ? ", " : "");
			}
			ss << "}";
			return ss.str();
		}
	};

	// 本年・累計成績情報
	class SEI_RUIKEI_INFO
	{
	public:
		std::string SetYear;		// 設定年
		std::string HonSyokinTotal; // 本賞金合計
		std::string FukaSyokin;		// 付加賞金合計
		CHAKUKAISU_INFO ChakuKaisu; // 着回数
		SEI_RUIKEI_INFO() : SetYear(""), HonSyokinTotal(""), FukaSyokin("") {}
		SEI_RUIKEI_INFO(std::string sei_ruikei_info)
		{
			if (sei_ruikei_info.size() != 60)
			{
				throw std::invalid_argument("Invalid SEI_RUIKEI_INFO format. Expected 60 characters.");
			}
			SetYear = sei_ruikei_info.substr(0, 4);
			HonSyokinTotal = sei_ruikei_info.substr(4, 10);
			FukaSyokin = sei_ruikei_info.substr(14, 10);
			ChakuKaisu = CHAKUKAISU_INFO(sei_ruikei_info.substr(24, 36));
		}

		std::string toString() const
		{
			std::stringstream ss;
			ss << "SEI_RUIKEI_INFO{SetYear=" << SetYear << ", HonSyokinTotal=" << HonSyokinTotal << ", FukaSyokin=" << FukaSyokin << ", " << ChakuKaisu.toString() << "}";
			return ss.str();
		}
	};

	// 最近重賞勝利情報
	class SAIKIN_JYUSYO_INFO
	{
	public:
		RACE_ID SaikinJyusyoid; // 年月日場回日R
		std::string Hondai;		// 競走名本題
		std::string Ryakusyo10; // 競走名略称10字
		std::string Ryakusyo6;	// 競走名略称6字
		std::string Ryakusyo3;	// 競走名略称3字
		std::string GradeCD;	// グレードコード
		std::string SyussoTosu; // 出走頭数
		std::string KettoNum;	// 血統登録番号
		std::string Bamei;		// 馬名
		SAIKIN_JYUSYO_INFO() : Hondai(""), Ryakusyo10(""), Ryakusyo6(""), Ryakusyo3(""), GradeCD(""), SyussoTosu(""), KettoNum(""), Bamei("") {}
		SAIKIN_JYUSYO_INFO(std::string saikin_jyusyo_info)
		{
			if (saikin_jyusyo_info.size() != 163)
			{
				throw std::invalid_argument("Invalid SAIKIN_JYUSYO_INFO format. Expected 163 characters.");
			}
			SaikinJyusyoid = RACE_ID(saikin_jyusyo_info.substr(0, 16));
			Hondai = saikin_jyusyo_info.substr(16, 60);
			Ryakusyo10 = saikin_jyusyo_info.substr(76, 20);
			Ryakusyo6 = saikin_jyusyo_info.substr(96, 12);
			Ryakusyo3 = saikin_jyusyo_info.substr(108, 6);
			GradeCD = saikin_jyusyo_info.substr(114, 1);
			SyussoTosu = saikin_jyusyo_info.substr(115, 2);
			KettoNum = saikin_jyusyo_info.substr(117, 10);
			Bamei = saikin_jyusyo_info.substr(127, 36);
		}

		std::string toString() const
		{
			std::stringstream ss;
			ss << "SAIKIN_JYUSYO_INFO{SaikinJyusyoid=" << SaikinJyusyoid.toString() << ", Hondai=" << Hondai << ", Ryakusyo10=" << Ryakusyo10 << ", Ryakusyo6=" << Ryakusyo6 << ", Ryakusyo3=" << Ryakusyo3 << ", GradeCD=" << GradeCD << ", SyussoTosu=" << SyussoTosu << ", KettoNum=" << KettoNum << ", Bamei=" << Bamei << "}";
			return ss.str();
		}
	};

	// 本年・前年・累計成績情報
	class HON_ZEN_RUIKEISEI_INFO
	{
	public:
		std::string SetYear;							// 設定年
		std::string HonSyokinHeichi;					// 平地本賞金合計
		std::string HonSyokinSyogai;					// 障害本賞金合計
		std::string FukaSyokinHeichi;					// 平地付加賞金合計
		std::string FukaSyokinSyogai;					// 障害付加賞金合計
		CHAKUKAISU_INFO ChakuKaisuHeichi;				// 平地着回数
		CHAKUKAISU_INFO ChakuKaisuSyogai;				// 障害着回数
		std::array<CHAKUKAISU_INFO, 20> ChakuKaisuJyo;	// 競馬場別着回数
		std::array<CHAKUKAISU_INFO, 6> ChakuKaisuKyori; // 距離別着回数
		HON_ZEN_RUIKEISEI_INFO() : SetYear(""), HonSyokinHeichi(""), HonSyokinSyogai(""), FukaSyokinHeichi(""), FukaSyokinSyogai("") {}
		HON_ZEN_RUIKEISEI_INFO(std::string hon_zen_ruikei_info)
		{
			if (hon_zen_ruikei_info.size() != 1052)
			{
				throw std::invalid_argument("Invalid HON_ZEN_RUIKEISEI_INFO format. Expected 1052 characters.");
			}
			SetYear = hon_zen_ruikei_info.substr(0, 4);
			HonSyokinHeichi = hon_zen_ruikei_info.substr(4, 10);
			HonSyokinSyogai = hon_zen_ruikei_info.substr(14, 10);
			FukaSyokinHeichi = hon_zen_ruikei_info.substr(24, 10);
			FukaSyokinSyogai = hon_zen_ruikei_info.substr(34, 10);
			ChakuKaisuHeichi = CHAKUKAISU_INFO(hon_zen_ruikei_info.substr(44, 36));
			ChakuKaisuSyogai = CHAKUKAISU_INFO(hon_zen_ruikei_info.substr(80, 36));
			for (int i = 0; i < 20; ++i)
			{
				ChakuKaisuJyo[i] = CHAKUKAISU_INFO(hon_zen_ruikei_info.substr(116 + i * 36, 36));
			}
			for (int i = 0; i < 6; ++i)
			{
				ChakuKaisuKyori[i] = CHAKUKAISU_INFO(hon_zen_ruikei_info.substr(836 + i * 36, 36));
			}
		}

		std::string toString() const
		{
			std::stringstream ss;
			ss << "HON_ZEN_RUIKEISEI_INFO{SetYear=" << SetYear << ", HonSyokinHeichi=" << HonSyokinHeichi << ", HonSyokinSyogai=" << HonSyokinSyogai << ", FukaSyokinHeichi=" << FukaSyokinHeichi << ", FukaSyokinSyogai=" << FukaSyokinSyogai << ", " << ChakuKaisuHeichi.toString() << ", " << ChakuKaisuSyogai.toString() << "}";
			for (size_t i = 0; i < ChakuKaisuJyo.size(); ++i)
			{
				ss << ", " << ChakuKaisuJyo[i].toString();
			}
			for (size_t i = 0; i < ChakuKaisuKyori.size(); ++i)
			{
				ss << ", " << ChakuKaisuKyori[i].toString();
			}
			ss << "}";
			return ss.str();
		}
	};

	// レース情報
	class RACE_INFO
	{
	public:
		std::string YoubiCD;	// 曜日コード
		std::string TokuNum;	// 特別競走番号
		std::string Hondai;		// 競走名本題
		std::string Fukudai;	// 競走名副題
		std::string Kakko;		// 競走名カッコ内
		std::string HondaiEng;	// 競走名本題欧字
		std::string FukudaiEng; // 競走名副題欧字
		std::string KakkoEng;	// 競走名カッコ内欧字
		std::string Ryakusyo10; // 競走名略称10字
		std::string Ryakusyo6;	// 競走名略称6字
		std::string Ryakusyo3;	// 競走名略称3字
		std::string Kubun;		// 競走名区分
		std::string Nkai;		// 重賞回次[第N回]
		RACE_INFO() : YoubiCD(""), TokuNum(""), Hondai(""), Fukudai(""), Kakko(""), HondaiEng(""), FukudaiEng(""), KakkoEng(""), Ryakusyo10(""), Ryakusyo6(""), Ryakusyo3(""), Kubun(""), Nkai("") {}
		RACE_INFO(std::string race_info)
		{
			if (race_info.size() != 587)
			{
				throw std::invalid_argument("Invalid RACE_INFO format. Expected 587 characters.");
			}
			YoubiCD = race_info.substr(0, 1);
			TokuNum = race_info.substr(1, 4);
			Hondai = race_info.substr(5, 60);
			Fukudai = race_info.substr(65, 60);
			Kakko = race_info.substr(125, 60);
			HondaiEng = race_info.substr(185, 120);
			FukudaiEng = race_info.substr(305, 120);
			KakkoEng = race_info.substr(425, 120);
			Ryakusyo10 = race_info.substr(545, 20);
			Ryakusyo6 = race_info.substr(565, 12);
			Ryakusyo3 = race_info.substr(577, 6);
			Kubun = race_info.substr(583, 1);
			Nkai = race_info.substr(584, 3);
		}

		std::string toString() const
		{
			std::stringstream ss;
			ss << "RACE_INFO{YoubiCD=" << YoubiCD << ", TokuNum=" << TokuNum << ", Hondai=" << Hondai << ", Fukudai=" << Fukudai << ", Kakko=" << Kakko << ", HondaiEng=" << HondaiEng << ", FukudaiEng=" << FukudaiEng << ", KakkoEng=" << KakkoEng << ", Ryakusyo10=" << Ryakusyo10 << ", Ryakusyo6=" << Ryakusyo6 << ", Ryakusyo3=" << Ryakusyo3 << ", Kubun=" << Kubun << ", Nkai=" << Nkai << "}";
			return ss.str();
		}
	};

	// 天候・馬場状態
	class TENKO_BABA_INFO
	{
	public:
		std::string TenkoCD;	// 天候コード
		std::string SibaBabaCD; // 芝馬場状態コード
		std::string DirtBabaCD; // ダート馬場状態コード
		TENKO_BABA_INFO() : TenkoCD(""), SibaBabaCD(""), DirtBabaCD("") {}
		TENKO_BABA_INFO(std::string tenko_baba_info)
		{
			if (tenko_baba_info.size() != 3)
			{
				throw std::invalid_argument("Invalid TENKO_BABA_INFO format. Expected 3 characters.");
			}
			TenkoCD = tenko_baba_info.substr(0, 1);
			SibaBabaCD = tenko_baba_info.substr(1, 1);
			DirtBabaCD = tenko_baba_info.substr(2, 1);
		}

		std::string toString() const
		{
			std::stringstream ss;
			ss << "TENKO_BABA_INFO{TenkoCD=" << TenkoCD << ", SibaBabaCD=" << SibaBabaCD << ", DirtBabaCD=" << DirtBabaCD << "}";
			return ss.str();
		}
	};

	// 競走条件コード
	class RACE_JYOKEN
	{
	public:
		std::string SyubetuCD;				 // 競走種別コード
		std::string KigoCD;					 // 競走記号コード
		std::string JyuryoCD;				 // 重量種別コード
		std::array<std::string, 5> JyokenCD; // 競走条件コード
		RACE_JYOKEN() : SyubetuCD(""), KigoCD(""), JyuryoCD("")
		{
			for (int i = 0; i < 5; ++i)
			{
				JyokenCD[i] = "";
			}
		}
		RACE_JYOKEN(std::string race_jyoken)
		{
			if (race_jyoken.size() != 21)
			{
				throw std::invalid_argument("Invalid RACE_JYOKEN format. Expected 21 characters.");
			}
			SyubetuCD = race_jyoken.substr(0, 2);
			KigoCD = race_jyoken.substr(2, 3);
			JyuryoCD = race_jyoken.substr(5, 1);
			for (int i = 0; i < 5; ++i)
			{
				JyokenCD[i] = race_jyoken.substr(6 + i * 3, 3);
			}
		}

		std::string toString() const
		{
			std::stringstream ss;
			ss << "RACE_JYOKEN{SyubetuCD=" << SyubetuCD << ", KigoCD=" << KigoCD << ", JyuryoCD=" << JyuryoCD << ", ";
			for (size_t i = 0; i < JyokenCD.size(); ++i)
			{
				ss << "JyokenCD[" << i << "]=" << JyokenCD[i] << (i < JyokenCD.size() - 1 ? ", " : "");
			}
			ss << "}";
			return ss.str();
		}
	};

	// 騎手変更情報
	class JC_INFO
	{
	public:
		std::string Futan;	   // 負担重量
		std::string KisyuCode; // 騎手コード
		std::string KisyuName; // 騎手名
		std::string MinaraiCD; // 騎手見習コード
		JC_INFO() : Futan(""), KisyuCode(""), KisyuName(""), MinaraiCD("") {}
		JC_INFO(std::string jc_info)
		{
			if (jc_info.size() != 43)
			{
				throw std::invalid_argument("Invalid JC_INFO format. Expected 43 characters.");
			}
			Futan = jc_info.substr(0, 3);
			KisyuCode = jc_info.substr(3, 5);
			KisyuName = jc_info.substr(8, 34);
			MinaraiCD = jc_info.substr(42, 1);
		}

		std::string toString() const
		{
			std::stringstream ss;
			ss << "JC_INFO{Futan=" << Futan << ", KisyuCode=" << KisyuCode << ", KisyuName=" << KisyuName << ", MinaraiCD=" << MinaraiCD << "}";
			return ss.str();
		}
	};

	// 発走時刻変更情報
	class TC_INFO
	{
	public:
		std::string Ji;	 // 時
		std::string Fun; // 分
		TC_INFO() : Ji(""), Fun("") {}
		TC_INFO(std::string tc_info)
		{
			if (tc_info.size() != 4)
			{
				throw std::invalid_argument("Invalid TC_INFO format. Expected 4 characters.");
			}
			Ji = tc_info.substr(0, 2);
			Fun = tc_info.substr(2, 2);
		}

		std::string toString() const
		{
			std::stringstream ss;
			ss << "TC_INFO{Ji=" << Ji << ", Fun=" << Fun << "}";
			return ss.str();
		}
	};

	// コース変更情報
	class CC_INFO
	{
	public:
		std::string Kyori;	 // 距離
		std::string TruckCD; // トラックコード
		CC_INFO() : Kyori(""), TruckCD("") {}
		CC_INFO(std::string cc_info)
		{
			if (cc_info.size() != 6)
			{
				throw std::invalid_argument("Invalid CC_INFO format. Expected 6 characters.");
			}
			Kyori = cc_info.substr(0, 4);
			TruckCD = cc_info.substr(4, 2);
		}

		std::string toString() const
		{
			std::stringstream ss;
			ss << "CC_INFO{Kyori=" << Kyori << ", TruckCD=" << TruckCD << "}";
			return ss.str();
		}
	};

	//////////////////// データクラス ////////////////////

	// 1. 特別登録馬
	class JV_TK_TOKUUMA
	{
	public:
		class TOKUUMA_INFO
		{
		public:
			std::string Num;			  // 連番
			std::string KettoNum;		  // 血統登録番号
			std::string Bamei;			  // 馬名
			std::string UmaKigoCD;		  // 馬記号コード
			std::string SexCD;			  // 性別コード
			std::string TozaiCD;		  // 調教師東西所属コード
			std::string ChokyosiCode;	  // 調教師コード
			std::string ChokyosiRyakusyo; // 調教師名略称
			std::string Futan;			  // 負担重量
			std::string Koryu;			  // 交流区分
			TOKUUMA_INFO() : Num(""), KettoNum(""), Bamei(""), UmaKigoCD(""), SexCD(""), TozaiCD(""), ChokyosiCode(""), ChokyosiRyakusyo(""), Futan(""), Koryu("") {}
			TOKUUMA_INFO(std::string tokuuma_info)
			{
				if (tokuuma_info.size() != 70)
				{
					throw std::invalid_argument("Invalid TOKUUMA_INFO format. Expected 70 characters.");
				}
				Num = tokuuma_info.substr(0, 3);
				KettoNum = tokuuma_info.substr(3, 10);
				Bamei = tokuuma_info.substr(13, 36);
				UmaKigoCD = tokuuma_info.substr(49, 2);
				SexCD = tokuuma_info.substr(51, 1);
				TozaiCD = tokuuma_info.substr(52, 1);
				ChokyosiCode = tokuuma_info.substr(53, 5);
				ChokyosiRyakusyo = tokuuma_info.substr(58, 8);
				Futan = tokuuma_info.substr(66, 3);
				Koryu = tokuuma_info.substr(69, 1);
			}

			std::string toString() const
			{
				std::stringstream ss;
				ss << "TOKUUMA_INFO{Num=" << Num << ", KettoNum=" << KettoNum << ", Bamei=" << Bamei << ", UmaKigoCD=" << UmaKigoCD << ", SexCD=" << SexCD << ", TozaiCD=" << TozaiCD << ", ChokyosiCode=" << ChokyosiCode << ", ChokyosiRyakusyo=" << ChokyosiRyakusyo << ", Futan=" << Futan << ", Koryu=" << Koryu << "}";
				return ss.str();
			}
		};

		RECORD_ID head;							   // レコードヘッダー
		RACE_ID id;								   // 競走識別情報
		RACE_INFO RaceInfo;						   // レース情報
		std::string GradeCD;					   // グレードコード
		RACE_JYOKEN JyokenInfo;					   // 競走条件コード
		std::string Kyori;						   // 距離
		std::string TrackCD;					   // トラックコード
		std::string CourseKubunCD;				   // コース区分
		YMD HandiDate;							   // ハンデ発表日
		std::string TorokuTosu;					   // 登録頭数
		std::array<TOKUUMA_INFO, 300> TokuUmaInfo; // 登録馬毎情報

		JV_TK_TOKUUMA() : head(), id(), RaceInfo(), GradeCD(""), JyokenInfo(), Kyori(""), TrackCD(""), CourseKubunCD(""), HandiDate(), TorokuTosu("")
		{
			for (int i = 0; i < 300; ++i)
			{
				TokuUmaInfo[i] = TOKUUMA_INFO();
			}
		}
		JV_TK_TOKUUMA(std::string jv_tk_tokuuma)
		{
			if (jv_tk_tokuuma.size() != 21655)
			{
				throw std::invalid_argument("Invalid JV_TK_TOKUUMA format. Expected 21655 characters.");
			}
			head = RECORD_ID(jv_tk_tokuuma.substr(0, 11));
			id = RACE_ID(jv_tk_tokuuma.substr(11, 16));
			RaceInfo = RACE_INFO(jv_tk_tokuuma.substr(27, 587));
			GradeCD = jv_tk_tokuuma.substr(614, 1);
			JyokenInfo = RACE_JYOKEN(jv_tk_tokuuma.substr(615, 21));
			Kyori = jv_tk_tokuuma.substr(636, 4);
			TrackCD = jv_tk_tokuuma.substr(640, 2);
			CourseKubunCD = jv_tk_tokuuma.substr(642, 2);
			HandiDate = YMD(jv_tk_tokuuma.substr(644, 8));
			TorokuTosu = jv_tk_tokuuma.substr(652, 3);
			for (int i = 0; i < 300; ++i)
			{
				TokuUmaInfo[i] = TOKUUMA_INFO(jv_tk_tokuuma.substr(655 + i * 70, 70));
			}
		}

		std::string toString() const
		{
			std::stringstream ss;
			ss << "JV_TK_TOKUUMA{head=" << head.toString() << ", id=" << id.toString() << ", RaceInfo=" << RaceInfo.toString() << ", GradeCD=" << GradeCD << ", JyokenInfo=" << JyokenInfo.toString() << ", Kyori=" << Kyori << ", TrackCD=" << TrackCD << ", CourseKubunCD=" << CourseKubunCD << ", HandiDate=" << HandiDate.toString() << ", TorokuTosu=" << TorokuTosu;
			for (size_t i = 0; i < TokuUmaInfo.size(); ++i)
			{
				ss << ", TokuUmaInfo[" << i << "]=" << TokuUmaInfo[i].toString();
			}
			ss << "}";
			return ss.str();
		}
	};

	// 2. レース詳細
	class JV_RA_RACE
	{
	public:
		class CORNER_INFO
		{
		public:
			std::string Corner;	  // コーナー
			std::string Syukaisu; // 周回数
			std::string Jyuni;	  // 各通過順位

			CORNER_INFO() : Corner(""), Syukaisu(""), Jyuni("") {}
			CORNER_INFO(std::string corner_info)
			{
				if (corner_info.size() != 72)
				{
					throw std::invalid_argument("Invalid CORNER_INFO format. Expected 72 characters.");
				}
				Corner = corner_info.substr(0, 1);
				Syukaisu = corner_info.substr(1, 1);
				Jyuni = corner_info.substr(2, 70);
			}

			std::string toString() const
			{
				std::stringstream ss;
				ss << "CORNER_INFO{Corner=" << Corner << ", Syukaisu=" << Syukaisu << ", Jyuni=" << Jyuni << "}";
				return ss.str();
			}
		};

		RECORD_ID head;								 // レコードヘッダー
		RACE_ID id;									 // 競走識別情報
		RACE_INFO RaceInfo;							 // レース情報
		std::string GradeCD;						 // グレードコード
		std::string GradeCDBefore;					 // 変更前グレードコード
		RACE_JYOKEN JyokenInfo;						 // 競走条件コード
		std::string JyokenName;						 // 競走条件名称
		std::string Kyori;							 // 距離
		std::string KyoriBefore;					 // 変更前距離
		std::string TrackCD;						 // トラックコード
		std::string TrackCDBefore;					 // 変更前トラックコード
		std::string CourseKubunCD;					 // コース区分
		std::string CourseKubunCDBefore;			 // 変更前コース区分
		std::array<std::string, 7> Honsyokin;		 // 本賞金
		std::array<std::string, 5> HonsyokinBefore;	 // 変更前本賞金
		std::array<std::string, 5> Fukasyokin;		 // 付加賞金
		std::array<std::string, 3> FukasyokinBefore; // 変更前付加賞金
		std::string HassoTime;						 // 発走時刻
		std::string HassoTimeBefore;				 // 変更前発走時刻
		std::string TorokuTosu;						 // 登録頭数
		std::string SyussoTosu;						 // 出走頭数
		std::string NyusenTosu;						 // 入線頭数
		TENKO_BABA_INFO TenkoBaba;					 // 天候・馬場状態
		std::array<std::string, 25> LapTime;		 // ラップタイム
		std::string SyogaiMileTime;					 // 障害マイルタイム
		std::string HaronTimeS3;					 // 前3ハロンタイム
		std::string HaronTimeS4;					 // 前4ハロンタイム
		std::string HaronTimeL3;					 // 後3ハロンタイム
		std::string HaronTimeL4;					 // 後4ハロンタイム
		std::array<CORNER_INFO, 4> CornerInfo;		 // コーナー通過順位
		std::string RecordUpKubun;					 // レコード更新区分

		JV_RA_RACE() : head(), id(), RaceInfo(), GradeCD(""), GradeCDBefore(""), JyokenInfo(), JyokenName(""), Kyori(""), KyoriBefore(""), TrackCD(""), TrackCDBefore(""), CourseKubunCD(""), CourseKubunCDBefore(""), HassoTime(""), HassoTimeBefore(""), TorokuTosu(""), SyussoTosu(""), NyusenTosu(""), TenkoBaba(), SyogaiMileTime(""), HaronTimeS3(""), HaronTimeS4(""), HaronTimeL3(""), HaronTimeL4(""), RecordUpKubun("")
		{
			for (int i = 0; i < 7; ++i)
			{
				Honsyokin[i] = "";
			}
			for (int i = 0; i < 5; ++i)
			{
				HonsyokinBefore[i] = "";
			}
			for (int i = 0; i < 5; ++i)
			{
				Fukasyokin[i] = "";
			}
			for (int i = 0; i < 3; ++i)
			{
				FukasyokinBefore[i] = "";
			}
			for (int i = 0; i < 25; ++i)
			{
				LapTime[i] = "";
			}
			for (int i = 0; i < 4; ++i)
			{
				CornerInfo[i] = CORNER_INFO();
			}
		}
		JV_RA_RACE(std::string jv_ra_race)
		{
			if (jv_ra_race.size() != 1270)
			{
				throw std::invalid_argument("Invalid JV_RA_RACE format. Expected 1270 characters.");
			}
			head = RECORD_ID(jv_ra_race.substr(0, 11));
			id = RACE_ID(jv_ra_race.substr(11, 16));
			RaceInfo = RACE_INFO(jv_ra_race.substr(27, 587));
			GradeCD = jv_ra_race.substr(614, 1);
			GradeCDBefore = jv_ra_race.substr(615, 1);
			JyokenInfo = RACE_JYOKEN(jv_ra_race.substr(616, 21));
			JyokenName = jv_ra_race.substr(637, 60);
			Kyori = jv_ra_race.substr(697, 4);
			KyoriBefore = jv_ra_race.substr(701, 4);
			TrackCD = jv_ra_race.substr(705, 2);
			TrackCDBefore = jv_ra_race.substr(707, 2);
			CourseKubunCD = jv_ra_race.substr(709, 2);
			CourseKubunCDBefore = jv_ra_race.substr(711, 2);
			for (int i = 0; i < 7; ++i)
			{
				Honsyokin[i] = jv_ra_race.substr(713 + i * 8, 8);
			}
			for (int i = 0; i < 5; ++i)
			{
				HonsyokinBefore[i] = jv_ra_race.substr(769 + i * 8, 8);
			}
			for (int i = 0; i < 5; ++i)
			{
				Fukasyokin[i] = jv_ra_race.substr(809 + i * 8, 8);
			}
			for (int i = 0; i < 3; ++i)
			{
				FukasyokinBefore[i] = jv_ra_race.substr(849 + i * 8, 8);
			}
			HassoTime = jv_ra_race.substr(873, 4);
			HassoTimeBefore = jv_ra_race.substr(877, 4);
			TorokuTosu = jv_ra_race.substr(881, 2);
			SyussoTosu = jv_ra_race.substr(883, 2);
			NyusenTosu = jv_ra_race.substr(885, 2);
			TenkoBaba = TENKO_BABA_INFO(jv_ra_race.substr(887, 3));
			for (int i = 0; i < 25; ++i)
			{
				LapTime[i] = jv_ra_race.substr(890 + i * 3, 3);
			}
			SyogaiMileTime = jv_ra_race.substr(965, 4);
			HaronTimeS3 = jv_ra_race.substr(969, 3);
			HaronTimeS4 = jv_ra_race.substr(972, 3);
			HaronTimeL3 = jv_ra_race.substr(975, 3);
			HaronTimeL4 = jv_ra_race.substr(978, 3);
			for (int i = 0; i < 4; ++i)
			{
				CornerInfo[i] = CORNER_INFO(jv_ra_race.substr(981 + i * 72, 72));
			}
			RecordUpKubun = jv_ra_race.substr(1269, 1);
		}

		std::string toString() const
		{
			std::stringstream ss;
			ss << "JV_RA_RACE{head=" << head.toString() << ", id=" << id.toString() << ", RaceInfo=" << RaceInfo.toString() << ", GradeCD=" << GradeCD << ", GradeCDBefore=" << GradeCDBefore << ", JyokenInfo=" << JyokenInfo.toString() << ", JyokenName=" << JyokenName << ", Kyori=" << Kyori << ", KyoriBefore=" << KyoriBefore << ", TrackCD=" << TrackCD << ", TrackCDBefore=" << TrackCDBefore << ", CourseKubunCD=" << CourseKubunCD << ", CourseKubunCDBefore=" << CourseKubunCDBefore;
			for (size_t i = 0; i < Honsyokin.size(); ++i)
			{
				ss << ", Honsyokin[" << i << "]=" << Honsyokin[i];
			}
			for (size_t i = 0; i < HonsyokinBefore.size(); ++i)
			{
				ss << ", HonsyokinBefore[" << i << "]=" << HonsyokinBefore[i];
			}
			for (size_t i = 0; i < Fukasyokin.size(); ++i)
			{
				ss << ", Fukasyokin[" << i << "]=" << Fukasyokin[i];
			}
			for (size_t i = 0; i < FukasyokinBefore.size(); ++i)
			{
				ss << ", FukasyokinBefore[" << i << "]=" << FukasyokinBefore[i];
			}
			ss << ", HassoTime=" << HassoTime << ", HassoTimeBefore=" << HassoTimeBefore << ", TorokuTosu=" << TorokuTosu << ", SyussoTosu=" << SyussoTosu << ", NyusenTosu=" << NyusenTosu << ", TenkoBaba=" << TenkoBaba.toString() << ", ";
			for (size_t i = 0; i < LapTime.size(); ++i)
			{
				ss << "LapTime[" << i << "]=" << LapTime[i] << (i < LapTime.size() - 1 ? ", " : "");
			}
			ss << ", SyogaiMileTime=" << SyogaiMileTime << ", HaronTimeS3=" << HaronTimeS3 << ", HaronTimeS4=" << HaronTimeS4 << ", HaronTimeL3=" << HaronTimeL3 << ", HaronTimeL4=" << HaronTimeL4;
			for (size_t i = 0; i < CornerInfo.size(); ++i)
			{
				ss << ", CornerInfo[" << i << "]=" << CornerInfo[i].toString();
			}
			ss << ", RecordUpKubun=" << RecordUpKubun << "}";
			return ss.str();
		}
	};
}