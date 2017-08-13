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

void resize_05x_grayscale(uint8_t* dst, uint8_t* src, int H, int W)
{
	for (int y=0; y<H; y++) {
		for (int x=0; x<W; x++) {
			dst[y*W+x] =
				(src[(y+0)*2*W + 2*(x+0)] >> 2) +
				(src[(y+0)*2*W + 2*(x+1)] >> 2) +
				(src[(y+1)*2*W + 2*(x+0)] >> 2) +
				(src[(y+1)*2*W + 2*(x+1)] >> 2);
		}
	}
}

void resize_05x_grayscale_two_sources(uint8_t* dst, uint8_t* src1, uint8_t* src2, int H, int W)
{
	for (int y=0; y<H; y++) {
		for (int x=0; x<W; x++) {
			uint8_t v1 =
				(src1[(y+0)*2*W + 2*(x+0)] >> 2) +
				(src1[(y+0)*2*W + 2*(x+1)] >> 2) +
				(src1[(y+1)*2*W + 2*(x+0)] >> 2) +
				(src1[(y+1)*2*W + 2*(x+1)] >> 2);
			uint8_t v2 =
				(src2[(y+0)*2*W + 2*(x+0)] >> 2) +
				(src2[(y+0)*2*W + 2*(x+1)] >> 2) +
				(src2[(y+1)*2*W + 2*(x+0)] >> 2) +
				(src2[(y+1)*2*W + 2*(x+1)] >> 2);
			dst[y*W+x] = std::max(v1, v2); // TODO: use some crazy http://www.geeksforgeeks.org/compute-the-minimum-or-maximum-max-of-two-integers-without-branching/
		}
	}
}

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
	std::vector< std::vector<uint8_t> > picture_stack;
	int picture_rot;
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

	MemMap(const std::string& fn, int size):
		_len(0),
		fd(-1),
		d(0)
	{
		fd = open(fn.c_str(), O_RDWR);
		if (fd==-1)
			throw std::runtime_error(stdprintf("cannot open file '%s': %s", fn.c_str(), strerror(errno)));
		_len = sizeof(T)*size;
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
		if (size % (LUMP*NCPU*BUNCH)) {
			close(fd);
			throw std::runtime_error(stdprintf("cannot divide size=%i by LUMP*NCPU*BUNCH*STEPS for '%s'",
				size,
				fn.c_str()));
		}
		chunk = size / (LUMP*NCPU*BUNCH*STEPS);
	}

	~MemMap()
	{
		if (d)
			munmap(d, _len);
		if (fd!=-1)
			close(fd);
	}

	T* at(int l, int b, int cursor)
	{
		return d + chunk*(l*NCPU*BUNCH*STEPS + cpu*BUNCH*STEPS + b*STEPS + cursor);
	}
	// shape_with_details    = [LUMP, NENV, STEPS]
	// NENV = NCPU*BUNCH
};

void main_loop()
{
	char buf[1024];
	snprintf(buf, sizeof(buf), "%i", cpu);
        FILE* monitor_js = fopen((monitor_dir + stdprintf("/%03i.monitor.json", cpu)).c_str(), "wt");
        double t0 = time();
        fprintf(monitor_js, "{\"t_start\": %0.2lf, \"gym_version\": \"vecgym\", \"env_id\": \"%s\"}\n", t0, env_id.c_str());
        fflush(monitor_js);

	MemMap<uint8_t> buf_obs0(prefix+"_obs0", LUMP*NCPU*BUNCH*STEPS*H*W*STACK);
        MemMap<int32_t> buf_acts(prefix+"_acts", LUMP*NCPU*BUNCH*STEPS);
        MemMap<float>   buf_rews(prefix+"_rews", LUMP*NCPU*BUNCH*STEPS);
        MemMap<bool>    buf_news(prefix+"_news", LUMP*NCPU*BUNCH*STEPS);
        MemMap<int32_t> buf_step(prefix+"_step", LUMP*NCPU*BUNCH*STEPS);
        MemMap<float>   buf_scor(prefix+"_scor", LUMP*NCPU*BUNCH*STEPS);

        MemMap<uint8_t> last_obs0(prefix+"_xlast_obs0", LUMP*NCPU*BUNCH*1*H*W*STACK);
        MemMap<float>   last_rews(prefix+"_xlast_rews", LUMP*NCPU*BUNCH*1);
        MemMap<bool>    last_news(prefix+"_xlast_news", LUMP*NCPU*BUNCH*1);
        MemMap<int32_t> last_step(prefix+"_xlast_step", LUMP*NCPU*BUNCH*1);
        MemMap<float>   last_scor(prefix+"_xlast_scor", LUMP*NCPU*BUNCH*1);

	std::vector<std::vector<ALEInterface*> > lumps;
	std::vector<std::vector<UsefulData> > lumps_useful;
	std::vector<uint8_t> full_res_buf1(FULL_PICTURE_BYTES);
	std::vector<uint8_t> full_res_buf2(FULL_PICTURE_BYTES);
	std::vector<Action> action_set;
	int cursor = 0;
	for (int l=0; l<LUMP; l++) {
		std::vector<ALEInterface*> bunch;
		std::vector<UsefulData> bunch_useful;
		for (int b=0; b<BUNCH; b++) {
			ALEInterface* emu = new ALEInterface();
			emu->setInt("random_seed", cpu*1000 + b);
			emu->loadROM(rom);
			action_set = emu->getMinimalActionSet();
			assert( FULL_PICTURE_BYTES == emu->getScreen().height() * emu->getScreen().width() );
			UsefulData data;
			data.frame = 0;
			data.score = 0;
			data.lives = emu->lives();
			for (int s=0; s<STACK; s++)
				data.picture_stack.push_back(std::vector<uint8_t>(W*H));
			emu->getScreenGrayscale(full_res_buf1);
			resize_05x_grayscale(data.picture_stack[0].data(), full_res_buf1.data(), H, W);
			for (int s=1; s<STACK; s++) {
				memcpy(data.picture_stack[s].data(), data.picture_stack[0].data(), W*H);
				memcpy(buf_obs0.at(l,b,cursor) + s*W*H, data.picture_stack[0].data(), W*H);
			}
			data.picture_rot = 0;
			bunch.push_back(emu);
			bunch_useful.push_back(data);
			buf_rews.at(l,b,cursor)[0] = 0;
			buf_news.at(l,b,cursor)[0] = true;
			buf_step.at(l,b,cursor)[0] = 0;
			buf_scor.at(l,b,cursor)[0] = 0;
		}
		lumps.push_back(bunch);
		lumps_useful.push_back(bunch_useful);
	}

	ssize_t r0 = write(fd_c2p_w, "R", 1);
	assert(r0==1); // pipe must block until it can write, not return errors.
	for (int l=0; l<LUMP; l++) {
		char buf[1];
		buf[0] = 'a' + l;
		ssize_t r0 = write(fd_c2p_w, buf, 1);
		assert(r0==1);
	}
        cursor += 1;
        bool quit = false;
        while (!quit) {
		bool last = cursor==STEPS;
		for (int l=0; l<LUMP; l++) {
			std::vector<ALEInterface*>& bunch = lumps[l];
			std::vector<UsefulData>& bunch_useful = lumps_useful[l];
			char cmd[2];
			//printf(" * * * * cpu%i rcv!\n", cpu);
			ssize_t r1 = read(fd_p2c_r, cmd, 1);
			//printf(" * * * * cpu%i cmd='%c'\n", cpu, cmd[0]);
			if (r1 != 1) {
				fprintf(stderr, "ale_vecgym_executable cpu%02i quit because of closed pipe (1)\n", cpu);
				return;
			}
			if (cmd[0]=='Q') return; // quit, but silently
			if (cmd[0] >= 'A' && cmd[0] <= 'H') { // 'H' is 8 lumps supported (ABCDEFGH)
				cursor = 0;
				assert(cmd[0]=='A' && "you have synchronization problems"); // but actually, it only makes sense to send 'A' there
			} else if (cmd[0] >= 'a' && cmd[0] <= 'h') {
				assert(cmd[0]==char(97+l) && "you have synchronization problems");
			} else {
				fprintf(stderr, "ale_vecgym_executable cpu%02i something strange visible in a pipe: '%c', let's quit just in case...\n", cpu, cmd[0]);
				return;
			}
	                assert(cursor < STEPS || last);
			for (int b=0; b<BUNCH; b++) {
				ALEInterface* emu = bunch[b];
				UsefulData& data = bunch_useful[b];
				Action ale_action = action_set[buf_acts.at(l,b,cursor-1)[0]];
				bool done = false;
				int  rew = 0;
				for (int s=0; s<SKIP; s++) {
					int r = emu->act(ale_action);
					rew  += r;
					data.frame += 1;
					data.score += r;
					done |= emu->game_over();
					if (done) break;
					if (s==SKIP-1) emu->getScreenGrayscale(full_res_buf1);
					if (s==SKIP-2) emu->getScreenGrayscale(full_res_buf2);
				}
				bool reset_me = done;
				if (!done) {
					resize_05x_grayscale_two_sources(data.picture_stack[data.picture_rot].data(), full_res_buf1.data(), full_res_buf2.data(), H, W);
					data.picture_rot += 1;
					data.picture_rot %= STACK;
					for (int s=0; s<STACK; s++) {
						int rot = (data.picture_rot + s) % STACK;
						if (!last) {
							memcpy(buf_obs0.at(l,b,cursor) + s*W*H, data.picture_stack[rot].data(), W*H);
						} else {
							memcpy(last_obs0.at(l,b,0) + s*W*H, data.picture_stack[rot].data(), W*H);
						}
					}
				}
				int lives = emu->lives();
				done |= lives < data.lives && lives > 0;
				bool life_lost = lives < data.lives;
				if (life_lost) rew = -1;
				data.lives = lives;
				//if env.__frame >= limit

				if (!last) {
					buf_rews.at(l,b,cursor)[0] = rew;
					buf_news.at(l,b,cursor)[0] = done || cursor==0;
					buf_scor.at(l,b,cursor)[0] = reset_me ? 0 : data.score;
					buf_step.at(l,b,cursor)[0] = reset_me ? 0 : data.frame;
				} else {
					last_rews.at(l,b,0)[0] = rew;
					last_news.at(l,b,0)[0] = done;
					last_scor.at(l,b,0)[0] = reset_me ? 0 : data.score;
					last_step.at(l,b,0)[0] = reset_me ? 0 : data.frame;
				}
				if (1 && cpu==0 && b==0 && l==0) {
					//fprintf(stderr, "%c", cmd[0]);
					//fflush(stderr);
					fprintf(stderr, " %05i frame %06i/%06i lives %i act %i total rew %i done %i\n",
						cursor,
						data.frame, 0, lives, int(ale_action),
						data.score, done);
				}

				if (!reset_me) continue;
				fprintf(monitor_js, "{\"r\": %i, \"l\": %i, \"t\": %0.2lf} )\n",
					data.score, data.frame, time() - t0);
				fflush(monitor_js);
				data.frame = 0;
				data.score = 0;
				data.lives = 0;
				emu->reset_game();
				emu->getScreenGrayscale(full_res_buf1);
				resize_05x_grayscale(data.picture_stack[0].data(), full_res_buf1.data(), H, W);
				for (int s=1; s<STACK; s++) {
					memcpy(data.picture_stack[s].data(), data.picture_stack[0].data(), W*H);
					memcpy(buf_obs0.at(l,b,cursor) + s*W*H, data.picture_stack[0].data(), W*H);
				}
				data.picture_rot = 0;
				for (int s=0; s<STACK; s++) {
					int rot = (data.picture_rot + s) % STACK;
					if (!last) {
						memcpy(buf_obs0.at(l,b,cursor) + s*W*H, data.picture_stack[rot].data(), W*H);
					} else {
						memcpy(last_obs0.at(l,b,0) + s*W*H, data.picture_stack[rot].data(), W*H);
					}
				}
				// But keep the rewards, step, score
			}
			char buf[1];
			buf[0] = 'a' + l;
			ssize_t r2 = write(fd_c2p_w, buf, 1);
			//printf(" * * * * cpu%i SENT\n", cpu);
			if (r2 != 1) {
				fprintf(stderr, "ale_vecgym_executable cpu%02i quit because of closed pipe (2)\n", cpu);
				return;
			}
		}
		cursor += 1;
	}
	fclose(monitor_js);
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
