#include <ale_interface.hpp>
#include <vector>
#include <stdarg.h>
#include <stdint.h>
#include <time.h>
#include <errno.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

static std::string env_id;
static std::string rom;
static std::string monitor_dir;
static std::string prefix;
static int LUMP;
static int cpu;
static int NCPU;
static int BUNCH;
static int STEPS;
static int SKIP;
static int STACK;
const int W = 80;
const int H = 105;
const int SMALL_PICTURE_BYTES =   H*W;
const int FULL_PICTURE_BYTES  = 210*160; // AKA 2H*2W
static int fd_p2c_r;
static int fd_c2p_w;

struct FrameStacking {
	std::vector<uint8_t> rot;
	std::vector<uint8_t> small1;
	std::vector<uint8_t> small2;
	uint16_t grayscale_palette[256];

	void init(ALEInterface* emu)
	{
		rot.resize(W*H*STACK);
		small1.resize(W*H);
		small2.resize(W*H);
		assert(2*W==emu->getScreen().width());
		assert(2*H==emu->getScreen().height());
		ColourPalette& pal = emu->theOSystem->colourPalette();
		uint8_t buf123[256];
		for (int c=0; c<256; c++) buf123[c] = c;
		uint8_t rgb[256*3];
		pal.applyPaletteRGB(rgb, buf123, 256);
		for (int c=0; c<256; c++) {
			float gray = 0.299f*rgb[3*c+0] + 0.587f*rgb[3*c+1] + 0.114f*rgb[3*c+2];
			uint16_t gray16 = uint16_t(gray);
			if (gray16==256) gray16 = 255;
			grayscale_palette[c] = gray16;
		}
	}

	void render_small(ALEInterface* ale, uint8_t* dst)
	{
		uint8_t* indexed = ale->getScreen().getArray();
		for (int y=0; y<H; y++) {
			int W2 = W*2;
			for (int x=0; x<W; x++) {
				dst[y*W+x] = uint8_t((
					(grayscale_palette[indexed[y*2*W2+0  + 2*x+0]]) +
					(grayscale_palette[indexed[y*2*W2+0  + 2*x+1]]) +
					(grayscale_palette[indexed[y*2*W2+W2 + 2*x+0]]) +
					(grayscale_palette[indexed[y*2*W2+W2 + 2*x+1]])
					) >> 2);
			}
		}
	}

	void rotate_and_max_two_small()
	{
		memmove(rot.data(), rot.data()+1, rot.size()-1);
		for (int c=0; c<H*W; c++)
			rot[STACK*c+(STACK-1)] = std::max(small1[c], small2[c]);
	}

	void fill_with_small1()
	{
		for (int c=0; c<H*W; c++)
			for (int s=0; s<STACK; s++)
				rot[STACK*c + s] = small1[c];
	}
};

std::string stdprintf(const char* fmt, ...)
{
	char buf[32768];
	va_list ap;
	va_start(ap, fmt);
	vsnprintf(buf, sizeof(buf), fmt, ap);
	va_end(ap);
	buf[32768-1] = 0;
	return buf;
}

double time()
{
	struct timespec time;
	clock_gettime(CLOCK_REALTIME, &time); // you need macOS Sierra 10.12 for this
	return time.tv_sec + 0.000000001*time.tv_nsec;
}

struct UsefulData {
	FrameStacking picture_stack;
	int lives;
	int frame;
	int score;
};

template<class T>
class MemMap {
	int _len;
public:
	int fd;
	T* d;
	int chunk;
	int steps;

	MemMap(const std::string& fn, int size, int steps):
		_len(0),
		fd(-1),
		d(0),
		steps(steps)
	{
		fd = open(fn.c_str(), O_RDWR);
		if (fd==-1)
			throw std::runtime_error(stdprintf("cannot open file '%s': %s", fn.c_str(), strerror(errno)));
		_len = sizeof(T)*size*steps;
		int file_on_disk_size = lseek(fd, 0, SEEK_END);
		if (file_on_disk_size != _len) {
			close(fd);
			throw std::runtime_error(stdprintf("file on disk '%s' has size %i, but expected size is %i",
				fn.c_str(), file_on_disk_size, _len) );
		}
		d = (T*) mmap(0, _len, PROT_READ|PROT_WRITE, MAP_SHARED, fd, 0);
		if (d==MAP_FAILED) {
			close(fd);
			throw std::runtime_error(stdprintf("cannot mmap '%s': %s", fn.c_str(), strerror(errno)));
		}
		if (_len % (LUMP*NCPU*BUNCH*steps)) {
			close(fd);
			throw std::runtime_error(stdprintf("%s cannot divide _len=%i by LUMP*NCPU*BUNCH*steps=%i for '%s'",
				fn.c_str(),
				_len,
				LUMP*NCPU*BUNCH*steps,
				fn.c_str()));
		}
		chunk = _len / (LUMP*NCPU*BUNCH*steps*sizeof(T));
	}

	~MemMap()
	{
		if (d) munmap(d, _len);
		if (fd!=-1) close(fd);
	}

	T* at(int l, int b, int cursor)
	{
		return d + chunk*(l*NCPU*BUNCH*steps + cpu*BUNCH*steps + b*steps + cursor);
	}
	// shape_with_details    = [LUMP, NENV, STEPS]
	// NENV = NCPU*BUNCH
};

class MemMapRGB {
public:
	int fd;
	uint8_t* d;

	MemMapRGB(): fd(-1), d(0)  { }

	void map(const std::string& fn)
	{
		fd = open(fn.c_str(), O_RDWR);  // S_IRUSR|S_IWUSR|S_IRGRP);
		if (fd==-1)
			throw std::runtime_error(stdprintf("cannot open file '%s': %s", fn.c_str(), strerror(errno)));
		//int z = ftruncate(fd, FULL_PICTURE_BYTES*3);
		//if (z!=0)
		//	throw std::runtime_error(stdprintf("cannot ftruncate file '%s': %s", fn.c_str(), strerror(errno)));
		d = (uint8_t*) mmap(0, FULL_PICTURE_BYTES*3, PROT_READ|PROT_WRITE, MAP_SHARED, fd, 0);
		if (d==MAP_FAILED) {
			close(fd);
			throw std::runtime_error(stdprintf("cannot mmap '%s': %s", fn.c_str(), strerror(errno)));
		}
	}

	~MemMapRGB()
	{
		if (d) munmap(d, FULL_PICTURE_BYTES*3);
		if (fd!=-1) close(fd);
	}

	std::vector<unsigned char> pic1;
	std::vector<unsigned char> pic2;

	void flush()
	{
		if (!d) return;
		for (int c=0; c<FULL_PICTURE_BYTES*3; c++)
			d[c] = std::max(pic1[c], pic2[c]);
	}

	void render(ALEInterface* emu, std::vector<unsigned char>* pic)
	{
		if (!d) return;
		pic->resize(FULL_PICTURE_BYTES*3);
		emu->getScreenRGB(*pic);
	}
};

void main_loop()
{
	double t0 = time();
	FILE* monitor_js = 0;
	if (!monitor_dir.empty()) {
		std::string monitor_fn = monitor_dir + stdprintf("/%03i.monitor.json", cpu);
		//fprintf(stderr, "ale_vecgym_executable cpu%02i monitor: %s\n", cpu, monitor_fn.c_str());
		monitor_js = fopen(monitor_fn.c_str(), "wt");
	}
	if (monitor_js) {
		fprintf(monitor_js, "{\"t_start\": %0.2lf, \"gym_version\": \"vecgym\", \"env_id\": \"%s\"}\n", t0, env_id.c_str());
		fflush(monitor_js);
	}

	MemMap<uint8_t> buf_obs0(prefix+"_obs0", LUMP*NCPU*BUNCH*H*W*STACK, STEPS);
	MemMap<float>   buf_vo0( prefix+"_vo0",  LUMP*NCPU*BUNCH, STEPS);
	MemMap<int32_t> buf_acts(prefix+"_acts", LUMP*NCPU*BUNCH, STEPS);
	MemMap<float>   buf_rews(prefix+"_rews", LUMP*NCPU*BUNCH, STEPS);
	MemMap<bool>    buf_news(prefix+"_news", LUMP*NCPU*BUNCH, STEPS);
	MemMap<int32_t> buf_step(prefix+"_step", LUMP*NCPU*BUNCH, STEPS);
	MemMap<float>   buf_scor(prefix+"_scor", LUMP*NCPU*BUNCH, STEPS);

	MemMap<uint8_t> last_obs0(prefix+"_xlast_obs0", LUMP*NCPU*BUNCH*H*W*STACK, 1);
	MemMap<bool>    last_news(prefix+"_xlast_news", LUMP*NCPU*BUNCH, 1);
	MemMap<int32_t> last_step(prefix+"_xlast_step", LUMP*NCPU*BUNCH, 1);
	MemMap<float>   last_scor(prefix+"_xlast_scor", LUMP*NCPU*BUNCH, 1);

	MemMapRGB rgb;
	if (STEPS==1 && NCPU==1 && LUMP==1)
		rgb.map(prefix + "_RGB");

	std::vector<std::vector<ALEInterface*> > lumps;
	std::vector<std::vector<UsefulData> > lumps_useful;
	std::vector<Action> action_set;
	int cursor = 0;
	for (int l=0; l<LUMP; l++) {
		std::vector<ALEInterface*> bunch;
		std::vector<UsefulData> bunch_useful;
		for (int b=0; b<BUNCH; b++) {
			ALEInterface* emu = new ALEInterface();
			emu->setInt("random_seed", cpu*1000 + b);
			emu->setFloat("repeat_action_probability", 0);
			emu->loadROM(rom);
			action_set = emu->getMinimalActionSet();
			assert( FULL_PICTURE_BYTES == emu->getScreen().height() * emu->getScreen().width() );
			UsefulData data;
			data.frame = 0;
			data.score = 0;
			data.lives = emu->lives();
			data.picture_stack.init(emu);
			data.picture_stack.render_small(emu, data.picture_stack.small1.data());
			data.picture_stack.fill_with_small1();
			rgb.render(emu, &rgb.pic1);
			rgb.render(emu, &rgb.pic2);
			bunch.push_back(emu);
			bunch_useful.push_back(data);
		}
		lumps.push_back(bunch);
		lumps_useful.push_back(bunch_useful);
	}
	//fprintf(stderr, "%s minimal action_set is %i long\n", env_id.c_str(), (int)action_set.size());

	ssize_t r0 = write(fd_c2p_w, "R", 1);
	assert(r0==1); // pipe must block until it can write, not return errors.

	char cmd[2];
	ssize_t r1 = read(fd_p2c_r, cmd, 1);
	if (r1 != 1) {
		fprintf(stderr, "ale_vecgym_executable cpu%02i quit because of closed pipe (1)\n", cpu);
		return;
	}
	assert(cmd[0]=='0' && "First command must be goto_buffer_beginning()");
	const int limit = 15000;

	bool quit = false;
	while (!quit) {
		assert(cursor==0);
		for (int l=0; l<LUMP; l++) {
			std::vector<UsefulData>& bunch_useful = lumps_useful[l];
			for (int b=0; b<BUNCH; b++) {
				UsefulData& data = bunch_useful[b];
				memcpy(buf_obs0.at(l,b,cursor), data.picture_stack.rot.data(), STACK*W*H);
				rgb.flush();
				buf_vo0.at(l,b,cursor)[0] = 1 - float(data.frame)/limit;
				buf_news.at(l,b,cursor)[0] = true;
				buf_step.at(l,b,cursor)[0] = data.frame;
				buf_scor.at(l,b,cursor)[0] = data.score;
			}
			char buf[1];
			buf[0] = 'a' + l;
			ssize_t r2 = write(fd_c2p_w, buf, 1);
			if (r2 != 1) {
				fprintf(stderr, "ale_vecgym_executable cpu%02i quit because of closed pipe (2)\n", cpu);
				return;
			}
		}

		int l = 0;
		while (!quit) {
			char cmd[2];
			//printf(" * * * * cpu%i rcv!\n", cpu);
			ssize_t r1 = read(fd_p2c_r, cmd, 1);
			//printf(" * * * * cpu%i cmd='%c'\n", cpu, cmd[0]);
			if (r1 != 1) {
				fprintf(stderr, "ale_vecgym_executable cpu%02i quit because of closed pipe (1)\n", cpu);
				return;
			}
			if (cmd[0]=='Q') {
				quit = true;
				break;
			}
			//if (cmd[0] >= 'A' && cmd[0] <= 'H') { // 'H' is 8 lumps supported (ABCDEFGH)
			//	cursor = 0;
			//assert(cmd[0]=='A' && "you have synchronization problems"); // but actually, it only makes sense to send 'A' there
			if (cmd[0] >= 'a' && cmd[0] <= 'h') {
				assert(cmd[0]==char(97+l) && "you have synchronization problems");
			} else if (cmd[0] == '0') {
				cursor = 0;
				break;
			} else {
				fprintf(stderr, "ale_vecgym_executable cpu%02i something strange visible in a pipe: '%c', let's quit just in case...\n", cpu, cmd[0]);
				quit = true;
				break;
			}

			std::vector<ALEInterface*>& bunch = lumps[l];
			std::vector<UsefulData>& bunch_useful = lumps_useful[l];
			assert(cursor < STEPS);
			for (int b=0; b<BUNCH; b++) {
				ALEInterface* emu = bunch[b];
				UsefulData& data = bunch_useful[b];
				int32_t a = buf_acts.at(l,b,cursor)[0];
				assert(a != 0xDEAD);
				Action ale_action = action_set[a];
				bool done = false;
				int  rew = 0;
				for (int s=0; s<SKIP; s++) {
					int r = emu->act(ale_action);
					rew  += r;
					data.frame += 1;
					data.score += r;
					done |= emu->game_over();
					if (done) break;
					if (s==SKIP-1) data.picture_stack.render_small(emu, data.picture_stack.small1.data());
					if (s==SKIP-2) data.picture_stack.render_small(emu, data.picture_stack.small2.data());
					if (s==SKIP-1) rgb.render(emu, &rgb.pic1);
					if (s==SKIP-2) rgb.render(emu, &rgb.pic2);
				}
				bool reset_me = done;
				int lives = emu->lives();
				done |= lives < data.lives && lives > 0;
				bool life_lost = lives < data.lives;
				if (life_lost) rew = -1;
				data.lives = lives;
				buf_rews.at(l,b,cursor)[0] = rew;
				if (data.frame >= limit)
					reset_me = true;

				if (0 && cpu==0 && b==0 && l==0) {
					//fprintf(stderr, "%c", cmd[0]);
					//fflush(stderr);
					fprintf(stderr, " %05i frame %06i/%06i lives %i act %i total rew %i done %i\n",
						cursor,
						data.frame, 0, lives, int(ale_action),
						data.score, done);
				}

				int save = cursor+1;
				if (!reset_me) {
					data.picture_stack.rotate_and_max_two_small();
					if (save < STEPS) {
						buf_news.at(l,b,save)[0] = done;
						buf_scor.at(l,b,save)[0] = data.score;
						buf_step.at(l,b,save)[0] = data.frame;
					} else {
						last_news.at(l,b,0)[0] = done;
						last_scor.at(l,b,0)[0] = data.score;
						last_step.at(l,b,0)[0] = data.frame;
					}
				} else {
					if (monitor_js) {
						fprintf(monitor_js, "{\"r\": %i, \"l\": %i, \"t\": %0.2lf}\n",
							data.score, data.frame, time() - t0);
						fflush(monitor_js);
					}
					data.frame = 0;
					data.score = 0;
					data.lives = 0;
					emu->reset_game();
					data.picture_stack.render_small(emu, data.picture_stack.small1.data());
					data.picture_stack.fill_with_small1();
					if (save < STEPS) {
						buf_news.at(l,b,save)[0] = true;
						buf_scor.at(l,b,save)[0] = data.score;
						buf_step.at(l,b,save)[0] = data.frame;
					} else {
						last_news.at(l,b,0)[0] = true;
						last_scor.at(l,b,0)[0] = data.score;
						last_step.at(l,b,0)[0] = data.frame;
					}
				}

				if (save < STEPS) {
					memcpy(buf_obs0.at(l,b,save), data.picture_stack.rot.data(), STACK*W*H);
					buf_vo0.at(l,b,save)[0] = 1 - float(data.frame)/limit;
				}  else {
					memcpy(last_obs0.at(l,b,0), data.picture_stack.rot.data(), STACK*W*H);
					buf_vo0.at(l,b,0)[0] = 1 - float(data.frame)/limit;
				}
				rgb.flush();
			}
			char buf[1];
			buf[0] = 'a' + l;
			ssize_t r2 = write(fd_c2p_w, buf, 1);
			//printf(" * * * * cpu%i SENT\n", cpu);
			if (r2 != 1) {
				fprintf(stderr, "ale_vecgym_executable cpu%02i quit because of closed pipe (2)\n", cpu);
				return;
			}

			l = (l+1) % LUMP;
			if (l==0)
			    cursor += 1;
		}
	}
	if (monitor_js) {
		fclose(monitor_js);
		monitor_js = 0;
	}
	close(fd_c2p_w);
	close(fd_p2c_r);
}

int main(int argc, char** argv)
{
	//ale::Logger::setMode(ale::Logger::Warning);
	ale::Logger::setMode(ale::Logger::Error);
	if (argc < 12) {
		fprintf(stderr, "I need more command line arguments!\n");
		return 1;
	}
	prefix      = argv[1];
	env_id      = argv[2];
	rom         = argv[3];
	monitor_dir = argv[4];
	LUMP    = atoi(argv[5]);
	cpu     = atoi(argv[6]);
	NCPU    = atoi(argv[7]);
	BUNCH   = atoi(argv[8]);
	STEPS   = atoi(argv[9]);
	SKIP    = atoi(argv[10]);
	STACK   = atoi(argv[11]);
	fd_p2c_r   = atoi(argv[12]);
	fd_c2p_w   = atoi(argv[13]);
	if (0) {
		fprintf(stderr, "\n*************************************** ALE VECGYM **************************************\n");
		fprintf(stderr, "C++ LUMP=%i cpu=%i/CPU=%i BUNCH=%i STEPS=%i SKIP=%i STACK=%i\n",
			LUMP,
			cpu,
			NCPU,
			BUNCH,
			STEPS,
			SKIP,
			STACK);
		fprintf(stderr, "        fds: p2c_r=%i c2p_w=%i\n", fd_p2c_r, fd_c2p_w);
		fprintf(stderr, "     prefix: %s\n", prefix.c_str());
		fprintf(stderr, "monitor dir: %s\n", monitor_dir.c_str());
		fprintf(stderr, "        rom: %s\n", rom.c_str());
	}
	main_loop();
	return 0;
}
