// Native implementation of this project's immutable-board search and public
// validator rules. No external solver source is incorporated. See README.md.
#include <algorithm>
#include <array>
#include <chrono>
#include <ctime>
#include <deque>
#include <iostream>
#include <map>
#include <memory>
#include <numeric>
#include <set>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#endif
using namespace std;
static double cpu_seconds(){
#ifdef _WIN32
 FILETIME creation,exit,kernel,user;
 if(GetProcessTimes(GetCurrentProcess(),&creation,&exit,&kernel,&user)){
  ULARGE_INTEGER k,u;k.LowPart=kernel.dwLowDateTime;k.HighPart=kernel.dwHighDateTime;u.LowPart=user.dwLowDateTime;u.HighPart=user.dwHighDateTime;
  return double(k.QuadPart+u.QuadPart)/10000000.0;
 }
#endif
 return double(clock())/CLOCKS_PER_SEC;
}
static const int INF=1000000000;
static const int rails[]={1,2,5,6,7,8,9,10,11,12,13,14,15,16};
static const int route[17][4]={
 {-1,-1,-1,-1},{0,1,-1,-1},{-1,-1,2,3},{-1,-1,-1,-1},{-1,-1,-1,-1},
 {2,-1,-1,1},{-1,2,-1,0},{3,-1,1,-1},{-1,3,0,-1},
 {2,1,-1,1},{2,-1,2,1},{0,2,-1,0},{-1,2,2,0},
 {3,1,1,-1},{3,-1,1,3},{0,3,0,-1},{-1,3,0,3}};
static int outgoing(int tile,int d){return tile>=0&&tile<17?route[tile][d]:-1;}
static int swapped(int t){switch(t){case 9:return 11;case 11:return 9;case 10:return 14;case 14:return 10;case 12:return 16;case 16:return 12;case 13:return 15;case 15:return 13;default:return t;}}
static bool simple(int t){return t==1||t==2||(t>=5&&t<=8);}
static bool junction(int t){return t>=9&&t<=16;}
static int tunnel_dir(int t){return t>=17&&t<=20?t-17:-1;}
static int goal_kind(int t){if(t==3||(t>=29&&t<=31))return 0;if(t==21||t==22||t==32||t==33)return 1;return -1;}
static int goal_dir(int t){switch(t){case 3:case 21:return 1;case 29:case 22:return 0;case 30:case 32:return 2;case 31:case 33:return 3;default:return -1;}}
static bool port(int tile,int d){return outgoing(tile,d^1)>=0;}
struct Car{int pos,dir,num,kind,status=0,dwell=0;bool stalled=false;};
// A checkpoint is a complete boundary before a synchronous tick. It is shared
// read-only by sibling layouts, never a partially computed set of proposals.
struct Checkpoint{
 vector<int> board,modifiers,entries,gate,pending;
 vector<Car> cars;
 vector<set<int>> visited;
 vector<vector<int>> visits;
 unordered_set<string> seen;
 array<int,2> delivered{{0,0}};
 int victory=-1,tick=0;
};
struct Layout{vector<int> domain,signal;int used=0,signals=0,depth=0;shared_ptr<const Checkpoint> checkpoint;};
struct Proposal{int pos,dir,entrance=-1,tunnel=-1;bool moved=false,deliver=false,wreck=false;};
struct Decision{int kind=-1,pos=-1,dir=-1,car=-1;bool prefer=false;vector<tuple<int,int,int>> connectors;vector<int> targets;};
struct Replay{bool success=false;int error=0,tick=0,start_tick=0;Decision query;shared_ptr<const Checkpoint> checkpoint;};
struct Engine{
 int h,w,n,budget=0,signal_budget=0,max_ticks=10000,lower_bound=0,best_cost=INF,best_signals=INF;
 bool optimize=true,unresolved=false,found=false,complete=false;
 int visit_cap=0,heuristic_passes=0;
 long long heuristic_nodes=0;
 long long checkpoint_resumes=0,reused_prefix_ticks=0,logical_ticks=0,full_replays=0;
 long long bound_cache_hits=0;
 long long node_limit=0;
 double timeout=0;
 int reason=0;
 vector<int> base,mods,nums,tunnels,mandatory,geometry,components,best_board,best_mods;
 vector<array<int,4>> adjacent;
 vector<vector<int>> graph,stations,goals;
 vector<vector<int>> route_predecessors;
 map<vector<int>,vector<int>> hint_cache;
 unordered_map<string,int> bound_cache;
 vector<Car> initial;
 array<long long,12> stats{};
 chrono::steady_clock::time_point started;
 double cpu_started;
 void prepare(){
  n=h*w;adjacent.resize(n);tunnels.assign(n,-1);geometry.assign(n,0);graph.resize(n);components.assign(n,-1);
  map<int,vector<int>> groups;
  for(int p=0;p<n;p++){
   adjacent[p]={{p%w?p-1:-1,p%w+1<w?p+1:-1,p/w+1<h?p+w:-1,p/w?p-w:-1}};
   if(tunnel_dir(base[p])>=0)groups[nums[p]].push_back(p);
  }
  for(auto &item:groups)if(item.second.size()==2){int a=item.second[0],b=item.second[1];tunnels[a]=b;tunnels[b]=a;}
  for(int p=0;p<n;p++){
   if(mods[p]==8)for(int d=0;d<4;d++)if(port(base[p],d)&&adjacent[p][d]>=0)geometry[adjacent[p][d]]=1;
   if(base[p]==4||(base[p]>=34&&base[p]<=37))continue;
   for(int q:adjacent[p])if(q>=0&&base[q]!=4&&!(base[q]>=34&&base[q]<=37))graph[p].push_back(q);
   if(tunnels[p]>=0)graph[p].push_back(tunnels[p]);
  }
  int comp=0;
  for(int p=0;p<n;p++)if(components[p]<0&&base[p]!=4&&!(base[p]>=34&&base[p]<=37)){
   vector<int> todo{p};components[p]=comp;
   while(!todo.empty()){int a=todo.back();todo.pop_back();for(int b:graph[a])if(components[b]<0){components[b]=comp;todo.push_back(b);}}
   comp++;
  }
  stations.resize(initial.size());goals.resize(initial.size());
  for(size_t i=0;i<initial.size();i++)if(initial[i].kind!=2)for(int p=0;p<n;p++){
   if(mods[p]==(initial[i].kind==0?6:11)&&nums[p]==initial[i].num)stations[i].push_back(p);
   if(goal_kind(base[p])==initial[i].kind)goals[i].push_back(p);
  }
  // Static directed distance fields are ordering hints only. They include all
  // possible new rails and both states of every switching junction.
  route_predecessors.resize(n*4);
  for(int p=0;p<n;p++)for(int d=0;d<4;d++){
   int q=adjacent[p][d];if(q<0)continue;int tile=base[q];
   auto edge=[&](int target){route_predecessors[target].push_back(p*4+d);};
   if(tunnel_dir(tile)>=0){if(d==(tunnel_dir(tile)^1)&&tunnels[q]>=0)edge(tunnels[q]*4+tunnel_dir(base[tunnels[q]]));continue;}
   if(goal_kind(tile)>=0)continue;
   int outputs=0;
   if(tile==0){for(int t:rails){int out=outgoing(t,d);if(out>=0)outputs|=1<<out;}}
   else{int out=outgoing(tile,d);if(out>=0)outputs|=1<<out;if(mods[q]==5||mods[q]==7){out=outgoing(swapped(tile),d);if(out>=0)outputs|=1<<out;}}
   for(int out=0;out<4;out++)if(outputs&(1<<out))edge(q*4+out);
  }
 }
 bool limit(){
  if(timeout>0&&chrono::duration<double>(chrono::steady_clock::now()-started).count()>=timeout){reason=3;return true;}
  if(node_limit>0&&stats[0]>=node_limit){reason=4;return true;}return false;
 }
 int bound(const Layout &a){
  stats[3]++;
  // This optimistic bound sees only occupied, explicitly empty and undecided
  // cells, just like the portable search's cache. Rail orientations, domain
  // refinements and semaphore decisions cannot change its value.
  string key;key.reserve(n);
  for(int p=0;p<n;p++)if(base[p]==0)key.push_back(a.domain[p]<0?0:a.domain[p]==0?1:2);
  auto cached=bound_cache.find(key);
  if(cached!=bound_cache.end()){bound_cache_hits++;return cached->second;}
  int value=compute_bound(a);
  if(bound_cache.size()>=8192)bound_cache.clear();
  bound_cache.emplace(move(key),value);return value;
 }
 int compute_bound(const Layout &a){
  int compulsory=0;vector<int> required(n,0);
  for(int p:mandatory)if(a.domain[p]<0){compulsory++;required[p]=1;}
  map<int,int> costs;
  for(size_t i=0;i<initial.size();i++)if(initial[i].kind!=2){
   vector<int> dist(n,INF);deque<int> todo;int start=initial[i].pos;dist[start]=0;todo.push_back(start);
   while(!todo.empty()){
    int p=todo.front();todo.pop_front();
    for(int q:graph[p])if(a.domain[q]!=0){
     int weight=base[q]==0&&a.domain[q]<0&&!required[q];int value=dist[p]+weight;
     if(value<dist[q]){dist[q]=value;if(weight)todo.push_back(q);else todo.push_front(q);}
    }
   }
   int goal=INF;for(int p:goals[i])goal=min(goal,dist[p]);int cost=goal;
   for(int p:stations[i])cost=max(cost,dist[p]);
   if(cost==INF)return INF;costs[components[start]]=max(costs[components[start]],cost);
  }
  int total=a.used+compulsory;for(auto item:costs)total+=item.second;return total;
 }
 int first(int mask) const {for(int t:rails)if(mask&(1<<t))return t;return 0;}
 void candidate(const Layout &a,vector<int>&board,vector<int>&modifiers){
  board=base;modifiers=mods;
  for(int p=0;p<n;p++){
   if(a.domain[p]>=0){int preferred=found?best_board[p]:0;board[p]=preferred&&(a.domain[p]&(1<<preferred))?preferred:first(a.domain[p]);}
   if(a.signal[p]==1)modifiers[p]=8;
  }
 }
 vector<tuple<int,int,int>> connectors(int p,const Layout&a){
  vector<tuple<int,int,int>> result;
  // A signal observes both its own physical rail and the connecting junction.
  // Its false branch preserves route domains. Its true branches specialize
  // only these observed cells, including an untraversed connecting junction.
  vector<int>hosts;
  if(base[p]){if(simple(base[p]))hosts.push_back(base[p]);}
  else for(int t:rails)if(simple(t)&&(a.domain[p]&(1<<t)))hosts.push_back(t);
  for(int host:hosts){
   vector<tuple<int,int,int>> choices;bool fixed=false;
   for(int d=0;d<4;d++)if(port(host,d)){
    int q=adjacent[p][d];if(q<0)continue;
    if(base[q]){if(junction(base[q])&&port(base[q],d^1))fixed=true;}
    else if(a.domain[q]>=0){for(int t=9;t<=16;t++)if((a.domain[q]&(1<<t))&&port(t,d^1)){
     if(a.domain[q]==(1<<t))fixed=true;else choices.push_back(make_tuple(host,q,t));
    }}else for(int t=9;t<=16;t++)if(port(t,d^1))choices.push_back(make_tuple(host,q,t));
   }
   if(fixed)result.push_back(make_tuple(host,-1,-1));
   else for(auto choice:choices)if(find(result.begin(),result.end(),choice)==result.end())result.push_back(choice);
  }return result;
 }
 Decision query(const vector<Car>&cars,const Layout&a,vector<set<int>>&visited){
  for(size_t i=0;i<cars.size();i++){
   const Car &c=cars[i];if(find(stations[i].begin(),stations[i].end(),c.pos)!=stations[i].end())visited[i].insert(c.pos);
   if(c.status>=2||c.dwell)continue;int p=adjacent[c.pos][c.dir];if(p<0)continue;
   Decision q;q.pos=p;q.dir=c.dir;q.car=int(i);
   for(int t:stations[i])if(!visited[i].count(t))q.targets.push_back(t);
   if(q.targets.empty())q.targets=goals[i];
   if(base[p]==0&&a.domain[p]<0){q.kind=0;return q;}
   if(a.domain[p]>0){int r=-2;for(int t:rails)if(a.domain[p]&(1<<t)){int out=outgoing(t,c.dir);if(r==-2)r=out;else if(r!=out){q.kind=1;return q;}}}
   if(a.signals<signal_budget&&a.signal[p]<0&&mods[p]==0){
    q.connectors=connectors(p,a);
    if(!q.connectors.empty()){
     q.kind=2;q.prefer=get<1>(q.connectors[0])<0&&any_of(cars.begin(),cars.end(),[&](const Car&other){return other.kind==c.kind&&other.num<c.num&&other.status!=2;});return q;
    }
   }
  }return Decision{};
 }
 string signature(const vector<int>&board,const vector<int>&m,const vector<Car>&cars,const vector<int>&entries,const vector<int>&gate,const vector<int>&pending,const int delivered[2],int victory){
  string key;key.reserve(n*3+cars.size()*28);
  auto add=[&](int value){for(int j=0;j<4;j++)key.push_back(char((unsigned(value)>>(j*8))&255));};
  for(int p=0;p<n;p++)if(mods[p]==5||mods[p]==7)add(board[p]);
  // Include all modifiers: new semaphores also change after release.
  for(int x:m)key.push_back(char(x));
  for(const Car&c:cars){add(c.pos);add(c.dir);add(c.status);add(c.dwell);add(c.stalled);}
  add(delivered[0]);add(delivered[1]);add(int(entries.size()));for(int x:entries)add(x);
  for(int p=0;p<n;p++)if(gate[p]>=0){add(gate[p]);add(pending[p]);}add(victory);return key;
 }
 bool directed_reachable(const vector<Car>&cars,const vector<int>&m,const Layout&a){
  // This is a necessary-condition relaxation, never a timing simulation.
  // Every unassigned rail permits every legal outgoing heading, every dynamic
  // junction permits both switch states, and all waits/collisions are ignored.
  // Thus every real future route occurs in this directed graph. We require
  // each outstanding stop and a correctly approached matching locomotive to
  // be reachable, but deliberately do not sum distances as a rail lower bound:
  // a real route can reuse a rail cell in different directions.
  for(size_t i=0;i<cars.size();i++){
   const Car&c=cars[i];if(c.kind==2||c.status==2)continue;
   vector<unsigned char>seen(n*4,0),reached(n,0);vector<int>todo{c.pos*4+c.dir};seen[todo[0]]=1;bool goal=false;
   while(!todo.empty()){
    int state=todo.back();todo.pop_back();int p=state/4,d=state%4,q=adjacent[p][d];if(q<0)continue;
    int tile=base[q];
    if(goal_kind(tile)>=0){if(goal_kind(tile)==c.kind&&goal_dir(tile)==d)goal=true;continue;}
    if(tunnel_dir(tile)>=0){if(d==(tunnel_dir(tile)^1)&&tunnels[q]>=0){int dest=tunnels[q]*4+tunnel_dir(base[tunnels[q]]);if(!seen[dest]){seen[dest]=1;todo.push_back(dest);}reached[tunnels[q]]=1;}continue;}
    int outputs=0;
    if(tile==0){
     if(a.domain[q]<0){for(int t:rails){int out=outgoing(t,d);if(out>=0)outputs|=1<<out;}}
     else for(int t:rails)if(a.domain[q]&(1<<t)){int out=outgoing(t,d);if(out>=0)outputs|=1<<out;}
    }else{
     int out=outgoing(tile,d);if(out>=0)outputs|=1<<out;
     if(mods[q]==5||mods[q]==7){out=outgoing(swapped(tile),d);if(out>=0)outputs|=1<<out;}
    }
    if(outputs)reached[q]=1;
    for(int out=0;out<4;out++)if(outputs&(1<<out)){int dest=q*4+out;if(!seen[dest]){seen[dest]=1;todo.push_back(dest);}}
   }
   if(!goal)return false;
   for(int p:stations[i])if(m[p]==(c.kind==0?6:11)&&!reached[p])return false;
  }return true;
 }
 Replay replay(vector<int>board,vector<int>m,const Layout&a){
  const Checkpoint*prefix=a.checkpoint.get();
  vector<Car> cars=prefix?prefix->cars:initial;
  vector<int>entries,gate,pending;
  vector<set<int>> visited;
  int delivered[2]={0,0},victory=-1;
  vector<vector<int>> visits;
  unordered_set<string> seen;
  Replay r;
  if(prefix){
   // Only original empty cells can change in a reusable child. They carry no
   // switching modifier. Keep every fixed dynamic rail and consumed modifier
   // from the prefix, and refresh the ordinary rail-domain representatives.
   vector<int>placed=move(board);board=prefix->board;m=prefix->modifiers;
   for(int p=0;p<n;p++)if(base[p]==0)board[p]=placed[p];
   entries=prefix->entries;gate=prefix->gate;pending=prefix->pending;
   visited=prefix->visited;visits=prefix->visits;seen=prefix->seen;
   delivered[0]=prefix->delivered[0];delivered[1]=prefix->delivered[1];victory=prefix->victory;
   r.start_tick=prefix->tick;r.tick=prefix->tick;
  }else{
   entries.resize(cars.size());iota(entries.begin(),entries.end(),0);gate.assign(n,-1);pending.assign(n,0);visited.resize(cars.size());
   if(visit_cap)visits.assign(cars.size(),vector<int>(n*4,0));
   for(int p=0;p<n;p++)if(m[p]==3||m[p]==4)gate[p]=m[p]==4;
   seen.insert(signature(board,m,cars,entries,gate,pending,delivered,victory));
  }
  // max_ticks remains an absolute tick bound, not a fresh allowance on resume.
  for(int tick=r.start_tick+1;tick<=max_ticks;tick++){
   if(visit_cap)for(size_t i=0;i<cars.size();i++){
    const Car&c=cars[i];if(c.kind!=2&&c.status<2&&!c.stalled&&!c.dwell&&++visits[i][c.pos*4+c.dir]>visit_cap){r.error=12;r.tick=tick-1;return r;}
   }
   r.query=query(cars,a,visited);if(r.query.kind>=0){
    r.tick=tick-1;
    if(!directed_reachable(cars,m,a)){r.query=Decision{};r.error=11;}
    else{
     r.error=10;
     auto saved=make_shared<Checkpoint>();
     saved->board=board;saved->modifiers=m;saved->cars=cars;saved->entries=entries;saved->gate=gate;saved->pending=pending;
     saved->delivered={{delivered[0],delivered[1]}};saved->victory=victory;saved->tick=r.tick;
     saved->visited=visited;saved->visits=visits;saved->seen=seen;
     // The query precedes movement and the visit-cap check runs before it.
     // Undo that check's count so a resumed child counts this tick once,
     // exactly as a fresh replay of the child's layout would do.
     if(visit_cap)for(size_t i=0;i<cars.size();i++){
      const Car&c=cars[i];if(c.kind!=2&&c.status<2&&!c.stalled&&!c.dwell)--saved->visits[i][c.pos*4+c.dir];
     }
     r.checkpoint=move(saved);
    }return r;
   }
   r.tick=tick;vector<int> occupied(n,0);for(const Car&c:cars)if(c.status!=2)occupied[c.pos]=1;
   for(int p=0;p<n;p++)if(pending[p]&&!occupied[p]){m[p]=3;pending[p]=0;}
   map<int,int>buttons;vector<pair<int,int>>entered;
   for(int i:entries){Car&c=cars[i];if(c.status>=2)continue;entered.push_back({c.pos,c.dir});if(m[c.pos]==1)buttons[nums[c.pos]]++;else if(m[c.pos]==7)board[c.pos]=swapped(board[c.pos]);}
   for(int p=0;p<n;p++){
    if(buttons[nums[p]]%2){
     if(gate[p]>=0){gate[p]=!gate[p];if(gate[p]){m[p]=4;pending[p]=0;}else if(occupied[p])pending[p]=1;else m[p]=3;}
     else if(m[p]==5)board[p]=swapped(board[p]);
    }
    if(m[p]==8)for(int d=0;d<4;d++)if(port(board[p],d)){
     int q=adjacent[p][d];if(q<0||!junction(board[q])||!port(board[q],d^1))continue;
     bool release=false;for(auto e:entered)if(e.first==q&&e.second!=(d^1)){release=true;break;}
     if(release){m[p]=9;break;}
    }
   }
   entries.clear();vector<Proposal> proposals;vector<pair<int,int>> failures;vector<int>stalled(cars.size(),0);
   for(size_t i=0;i<cars.size();i++){
    Car&c=cars[i];Proposal p;p.pos=c.pos;p.dir=c.dir;proposals.push_back(p);Proposal &v=proposals.back();stalled[i]=c.stalled;
    if(c.status>=2)continue;c.stalled=false;
    if(c.dwell){c.dwell--;c.status=c.dwell?1:0;continue;}
    int dest=adjacent[c.pos][c.dir];
    if(dest>=0&&(m[dest]==3||m[dest]==8)){c.stalled=true;c.status=1;continue;}
    c.status=0;
    if(dest<0){if(c.kind==2)v.wreck=true;else failures.push_back({1,int(i)});continue;}
    int tile=board[dest],kind=goal_kind(tile);
    if(kind>=0){
     int fail=0;
     if(c.dir!=goal_dir(tile)){if(c.kind==2)v.wreck=true;else fail=1;}
     else if(c.kind==2)fail=2;
     else if(c.kind!=kind)fail=3;
     else if(c.num!=delivered[kind])fail=4;
     else{for(int p=0;p<n;p++)if(m[p]==(kind==0?6:11)&&nums[p]==c.num){fail=5;break;}if(!fail){v.pos=dest;v.moved=true;v.deliver=true;}}
     if(fail)failures.push_back({fail,int(i)});continue;
    }
    if(tunnel_dir(tile)>=0){
     if(c.dir!=(tunnel_dir(tile)^1)){if(c.kind==2)v.wreck=true;else failures.push_back({1,int(i)});continue;}
     int q=tunnels[dest];if(q<0){r.error=1;return r;}v.pos=q;v.moved=true;v.entrance=dest;v.tunnel=min(dest,q);v.dir=tunnel_dir(board[q]);continue;
    }
    int out=outgoing(tile,c.dir);
    if(out<0){if(c.kind==2)v.wreck=true;else failures.push_back({1,int(i)});continue;}
    v.pos=dest;v.dir=out;v.moved=true;
   }
   set<int> blocked;
   for(size_t i=0;i<cars.size();i++)for(size_t j=i+1;j<cars.size();j++){
    const Proposal&a=proposals[i],&b=proposals[j];int first_a=a.entrance>=0?a.entrance:a.pos,first_b=b.entrance>=0?b.entrance:b.pos;
    if(a.moved&&b.moved&&first_a==cars[j].pos&&first_b==cars[i].pos){if(a.entrance>=0)blocked.insert(int(i));if(b.entrance>=0)blocked.insert(int(j));}
   }
   for(int i:blocked){auto&p=proposals[i];p.pos=p.entrance;p.dir=cars[i].dir;p.tunnel=-1;}
   while(true){
    map<int,int>collision_failures;set<int>wrecks;
    for(size_t i=0;i<cars.size();i++)if(cars[i].status!=2)for(size_t j=i+1;j<cars.size();j++)if(cars[j].status!=2){
     const Proposal&a=proposals[i],&b=proposals[j];bool tunnel=a.tunnel>=0&&a.tunnel==b.tunnel;
     bool collision=a.pos==b.pos||(a.moved&&b.moved&&a.pos==cars[j].pos&&b.pos==cars[i].pos)||(stalled[i]&&b.moved&&b.pos==cars[i].pos)||(stalled[j]&&a.moved&&a.pos==cars[j].pos)||(a.entrance>=0&&a.entrance==b.pos)||(b.entrance>=0&&b.entrance==a.pos);
     if(!collision&&!tunnel)continue;
     if(cars[i].kind==2&&cars[j].kind==2){if(!a.wreck)wrecks.insert(int(i));if(!b.wreck)wrecks.insert(int(j));}
     else{collision_failures[int(i)]=tunnel?7:6;collision_failures[int(j)]=tunnel?7:6;}
    }
    if(wrecks.empty()){for(auto item:collision_failures)failures.push_back({item.second,item.first});break;}
    for(int i:wrecks){Proposal&p=proposals[i];p.pos=cars[i].pos;p.moved=false;p.wreck=true;p.tunnel=-1;p.entrance=-1;}
   }
   for(size_t i=0;i<cars.size();i++){
    Car&c=cars[i];const Proposal&p=proposals[i];if(c.status==2)continue;
    if(p.wreck){c.status=3;c.stalled=false;c.dwell=0;}
    else if(p.moved){c.pos=p.pos;c.dir=p.dir;if(p.deliver){c.status=2;delivered[c.kind]++;}else{entries.push_back(int(i));if((c.kind==0&&m[c.pos]==6||c.kind==1&&m[c.pos]==11)&&nums[c.pos]==c.num){c.dwell=2;c.status=1;m[c.pos]=9;}}}
   }
   if(!failures.empty()){r.error=failures[0].first;return r;}
   bool all=true,decoy=false;for(const Car&c:cars){if(c.kind!=2&&c.status!=2)all=false;if(c.kind==2&&c.status!=3)decoy=true;}
   if(all)victory=victory<0?2:victory-1;
   if(all&&(victory==0||!decoy)){r.success=true;return r;}
   string key=signature(board,m,cars,entries,gate,pending,delivered,victory);
   if(!seen.insert(move(key)).second){r.error=8;return r;}
  }r.error=9;return r;
 }
 bool possible_next(int pos,int heading,const Layout&a,int kind){
  int p=adjacent[pos][heading];if(p<0)return false;
  if(base[p]==0&&a.domain[p]<0)return true;
  if(a.domain[p]>=0){for(int t:rails)if((a.domain[p]&(1<<t))&&outgoing(t,heading)>=0)return true;return false;}
  int t=base[p];if(outgoing(t,heading)>=0)return true;
  if(tunnel_dir(t)>=0)return heading==(tunnel_dir(t)^1);
  return goal_kind(t)==kind&&goal_dir(t)==heading;
 }
 bool signal_geometry(int pos,const Layout&a){
  if(geometry[pos])return true;
  for(int d=0;d<4;d++){
   int signal=adjacent[pos][d];
   if(signal>=0&&a.signal[signal]==1){int tile=base[signal]?base[signal]:first(a.domain[signal]);if(port(tile,d^1))return true;}
  }return false;
 }
 vector<int> options(const Decision&q,const Layout&a){
  int kind=initial[q.car].kind;vector<int> choices;int domain=a.domain[q.pos];
  if(kind==2&&domain<0)choices.push_back(0);
  for(int t:rails)if((domain<0||(domain&(1<<t)))&&(kind==2||(outgoing(t,q.dir)>=0&&possible_next(q.pos,outgoing(t,q.dir),a,kind))))choices.push_back(t);
  auto cached=hint_cache.find(q.targets);
  if(cached==hint_cache.end()){
   vector<int> distances(n*4,INF);deque<int>todo;
   for(int p:q.targets){
    if(goal_kind(base[p])>=0){int d=goal_dir(base[p]),previous=adjacent[p][d^1];if(previous>=0){distances[previous*4+d]=0;todo.push_back(previous*4+d);}}
    else for(int d=0;d<4;d++){distances[p*4+d]=0;todo.push_back(p*4+d);}
   }
   while(!todo.empty()){
    int p=todo.front();todo.pop_front();
    for(int other:route_predecessors[p]){
     int entered=adjacent[other/4][other%4];
     int weight=entered>=0&&base[entered]==0;
     int value=distances[p]+weight;
     if(value<distances[other]){
      distances[other]=value;
      if(weight)todo.push_back(other);else todo.push_front(other);
     }
    }
   }
   cached=hint_cache.emplace(q.targets,move(distances)).first;
  }
  const vector<int>&distances=cached->second;
  int preferred=found?best_board[q.pos]:-1;
  auto score=[&](int t){int out=outgoing(t,q.dir);int distance=out<0?INF:distances[q.pos*4+out];return make_tuple(kind==2&&t!=0,t!=preferred,distance,!simple(t),t);};
  stable_sort(choices.begin(),choices.end(),[&](int x,int y){return score(x)<score(y);});
  vector<int> result;vector<int> keys;
  bool observed_geometry=signal_geometry(q.pos,a);
  for(int t:choices){int key=t==0?-2:outgoing(t,q.dir);int mask=t==0?0:(1<<t);
   if(!observed_geometry){auto it=find(keys.begin(),keys.end(),key);if(it!=keys.end()){result[size_t(it-keys.begin())]|=mask;continue;}}
   keys.push_back(key);result.push_back(mask);
  }return result;
 }
 void search(long long phase_limit=0){
  long long phase_start=stats[0];
  Layout first;first.domain.assign(n,-1);first.signal.assign(n,-1);vector<Layout>stack{first};
  while(!stack.empty()){
   if(limit())return;if(phase_limit&&stats[0]-phase_start>=phase_limit){reason=6;return;}Layout a=move(stack.back());stack.pop_back();stats[0]++;stats[9]=max(stats[9],(long long)a.depth);
   int cost=bound(a);if(cost>budget||(optimize&&cost>=best_cost)){stats[4]++;continue;}
   vector<int>board,m;candidate(a,board,m);stats[1]++;Replay r=replay(board,m,a);
   stats[2]+=r.tick-r.start_tick;logical_ticks+=r.tick;
   if(a.checkpoint){checkpoint_resumes++;reused_prefix_ticks+=r.start_tick;}else full_replays++;
   if(r.query.kind>=0){
    const Decision&q=r.query;
    if(q.kind<2){stats[q.kind==0?6:7]++;vector<int>choices=options(q,a);
     for(auto it=choices.rbegin();it!=choices.rend();it++){
      Layout child=a;int used=a.used+(*it!=0)-(a.domain[q.pos]>0);if(used>budget)continue;
      child.domain[q.pos]=*it;child.used=used;child.depth+=a.domain[q.pos]<0;
      // Ordinary route-compatible changes preserve the whole observed prefix.
      // A physical signal-neighbor change conservatively replays from tick 0.
      child.checkpoint=signal_geometry(q.pos,a)?nullptr:r.checkpoint;
      stack.push_back(move(child));
     }
    }else{
     stats[8]++;vector<Layout>children;Layout no=a;no.signal[q.pos]=0;no.depth++;no.checkpoint=r.checkpoint;children.push_back(move(no));
     for(auto addition:q.connectors){Layout yes=a;yes.checkpoint.reset();yes.signal[q.pos]=1;yes.signals++;yes.depth++;
      if(base[q.pos]==0)yes.domain[q.pos]=1<<get<0>(addition);
      int neighbor=get<1>(addition);
      if(neighbor>=0){if(yes.domain[neighbor]<0){yes.used++;yes.depth++;}yes.domain[neighbor]=1<<get<2>(addition);}
      if(yes.used<=budget){
       // Every connected port can release this signal, not just the neighbor
       // selected to establish static legality. Fix all already observed
       // neighboring domains now; future assignments use concrete geometry.
       vector<Layout> geometries{move(yes)};
       int host=get<0>(addition);
       for(int d=0;d<4;d++)if(port(host,d)){
        int neighbor=adjacent[q.pos][d];if(neighbor<0||base[neighbor])continue;
        vector<Layout>expanded;
        for(const Layout&g:geometries){
         int mask=g.domain[neighbor];
         if(mask<=0||(mask&(mask-1))==0){expanded.push_back(g);continue;}
         for(int tile:rails)if(mask&(1<<tile)){Layout concrete=g;concrete.domain[neighbor]=1<<tile;expanded.push_back(move(concrete));}
        }geometries=move(expanded);
       }
       for(Layout&g:geometries)children.push_back(move(g));
      }
     }
     if(q.prefer)rotate(children.begin(),children.begin()+1,children.end());
     for(auto it=children.rbegin();it!=children.rend();it++)stack.push_back(move(*it));
    }continue;
   }
   if(r.success){
    stats[11]++;if(a.used<best_cost||(a.used==best_cost&&a.signals<best_signals)){best_cost=a.used;best_signals=a.signals;best_board=move(board);best_mods=move(m);found=true;}
    if(!optimize||best_cost==lower_bound){reason=optimize?2:1;complete=optimize;return;}
   }else if(r.error==9){unresolved=true;stats[10]++;}else stats[5]++;
  }reason=unresolved?5:0;complete=!unresolved&&!visit_cap;
 }
};
int main(){
 ios::sync_with_stdio(false);cin.tie(nullptr);
 int count;Engine e;e.cpu_started=cpu_seconds();if(!(cin>>e.h>>e.w>>count))return 2;
 if(e.h<1||e.w<1||e.h>65536||e.w>65536||e.h*e.w>65536)return 2;e.n=e.h*e.w;
 for(vector<int>*v:{&e.base,&e.mods,&e.nums}){v->resize(e.n);for(int &x:*v)cin>>x;}
 for(int i=0;i<count;i++){Car c;cin>>c.pos>>c.dir>>c.num>>c.kind;e.initial.push_back(c);}
 e.prepare();
 cin>>e.budget>>e.signal_budget>>e.max_ticks>>e.optimize>>e.node_limit>>e.timeout>>e.lower_bound>>e.best_cost;
 int required;cin>>required;for(int i=0,p;i<required;i++){cin>>p;e.mandatory.push_back(p);}
 cin>>e.found;if(e.found){e.best_board.resize(e.n);e.best_mods.resize(e.n);for(int &x:e.best_board)cin>>x;for(int &x:e.best_mods)cin>>x;}
 e.started=chrono::steady_clock::now();
 // Visit limits are explicitly a witness-finding heuristic. Failure here is
 // never a proof: progressively relax them, then restart the complete tree.
 for(int cap=1;cap<=3;cap++){
  e.visit_cap=cap;e.heuristic_passes++;long long before=e.stats[0];e.search(100000);e.heuristic_nodes+=e.stats[0]-before;
  if(e.complete||e.found&&!e.optimize||e.reason==3||e.reason==4)break;
  e.unresolved=false;
 }
 if(!e.complete&&!(e.found&&!e.optimize)&&e.reason!=3&&e.reason!=4){e.visit_cap=0;e.unresolved=false;e.search();}
 cout.precision(15);cout<<e.complete<<' '<<e.reason<<' '<<e.found<<' '<<cpu_seconds()-e.cpu_started;
 for(long long x:e.stats)cout<<' '<<x;
 cout<<' '<<e.heuristic_nodes<<' '<<e.heuristic_passes;
 cout<<' '<<e.checkpoint_resumes<<' '<<e.reused_prefix_ticks<<' '<<e.logical_ticks<<' '<<e.full_replays;
 cout<<' '<<e.bound_cache_hits;
 if(e.found){for(int x:e.best_board)cout<<' '<<x;for(int x:e.best_mods)cout<<' '<<x;}
 cout<<'\n';return 0;
}
