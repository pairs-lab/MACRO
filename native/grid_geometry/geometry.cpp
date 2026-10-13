#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <map>
#include <memory>
#include <queue>
#include <set>
#include <stdexcept>
#include <tuple>
#include <vector>

// No fast-math: heap ordering, strict relaxation and clearance epsilon must
// match the Python reference. Python owns candidate scoring and prediction.
using Cell = std::pair<long, long>;
struct Ref {
    PyObject* p;
    explicit Ref(PyObject* value): p(value) {}
    ~Ref() { Py_XDECREF(p); }
    Ref(const Ref&) = delete;
};
static void checked(bool ok) { if (!ok) throw std::runtime_error("invalid geometry input"); }
static Cell cell(PyObject* value) {
    Ref seq(PySequence_Fast(value, "expected a grid coordinate"));
    checked(seq.p && PySequence_Fast_GET_SIZE(seq.p) == 2);
    auto a = PySequence_Fast_GET_ITEM(seq.p, 0);
    auto b = PySequence_Fast_GET_ITEM(seq.p, 1);
    checked(PyLong_Check(a) && PyLong_Check(b));
    Cell result{PyLong_AsLong(a), PyLong_AsLong(b)};
    checked(!PyErr_Occurred());
    return result;
}
static PyObject* point(Cell p) { return Py_BuildValue("(ll)", p.first, p.second); }
static std::vector<Cell> cells(PyObject* value) {
    Ref iterator(PyObject_GetIter(value)); checked(iterator.p);
    std::vector<Cell> result;
    while (PyObject* next = PyIter_Next(iterator.p)) {
        Ref item(next); result.push_back(cell(item.p));
    }
    checked(!PyErr_Occurred()); return result;
}
static PyObject* points(const std::vector<Cell>& values) {
    Ref result(PyList_New(values.size())); checked(result.p);
    for (size_t i=0; i<values.size(); ++i) {
        PyObject* p = point(values[i]); checked(p);
        PyList_SET_ITEM(result.p, i, p);
    }
    return Py_NewRef(result.p);
}
static PyObject* failure() {
    if (!PyErr_Occurred()) PyErr_SetString(PyExc_ValueError, "invalid native geometry input");
    return nullptr;
}

using BodyKey = std::pair<std::vector<std::pair<double,double>>, double>;
using EdgeKey = std::tuple<long,long,long,long,uint64_t>;
struct BodyCache {
    std::map<BodyKey, std::map<EdgeKey,std::pair<bool,uint64_t>>> groups;
    size_t entries=0, hits=0, misses=0;
};
static const char* CACHE_NAME="marco.clearance.body.v2";
static void destroy_cache(PyObject* capsule) {
    delete static_cast<BodyCache*>(PyCapsule_GetPointer(capsule,CACHE_NAME));
}
static PyObject* new_cache(PyObject*,PyObject*) {
    auto* cache=new BodyCache();
    PyObject* result=PyCapsule_New(cache,CACHE_NAME,destroy_cache);
    if (!result) delete cache;
    return result;
}
static PyObject* cache_stats(PyObject*,PyObject* capsule) {
    auto* cache=static_cast<BodyCache*>(PyCapsule_GetPointer(capsule,CACHE_NAME));
    if (!cache) return nullptr;
    return Py_BuildValue("{sK,sK,sK}","hits",(unsigned long long)cache->hits,
                         "misses",(unsigned long long)cache->misses,
                         "entries",(unsigned long long)cache->entries);
}

struct GridNeighbor {Cell next, side1, side2; bool diagonal;};
struct NativeGrid {
    std::set<Cell> free;
    std::map<Cell,std::vector<GridNeighbor>> neighbors;
};
static const char* GRID_NAME="marco.navigation.grid.v1";
static void destroy_grid(PyObject* capsule) {
    delete static_cast<NativeGrid*>(PyCapsule_GetPointer(capsule,GRID_NAME));
}
static PyObject* prepare_grid(PyObject*,PyObject* object) {
    try {
        auto values=cells(object);
        auto* grid=new NativeGrid();
        // Own the allocation until the capsule can take responsibility.
        std::unique_ptr<NativeGrid> owner(grid);
        grid->free.insert(values.begin(),values.end());
        const long moves[][2]={{-1,0},{1,0},{0,-1},{0,1},{-1,-1},{-1,1},{1,-1},{1,1}};
        for (auto current:grid->free) {
            auto& adjacent=grid->neighbors[current];
            for (auto& move:moves) {
                long dx=move[0],dy=move[1];
                Cell next{current.first+dx,current.second+dy};
                Cell side1{current.first+dx,current.second},side2{current.first,current.second+dy};
                bool diagonal=dx && dy;
                if (!grid->free.count(next) ||
                    (diagonal && (!grid->free.count(side1) || !grid->free.count(side2)))) continue;
                adjacent.push_back({next,side1,side2,diagonal});
            }
        }
        PyObject* capsule=PyCapsule_New(grid,GRID_NAME,destroy_grid);
        checked(capsule); owner.release(); return capsule;
    } catch (...) {return failure();}
}

static PyObject* tree(PyObject*, PyObject* args) {
    PyObject *free_object, *start_object, *goal_object, *blocked_object;
    double limit;
    if (!PyArg_ParseTuple(args, "OOOOd", &free_object, &start_object,
                         &goal_object, &blocked_object, &limit)) return nullptr;
    try {
        NativeGrid* grid=nullptr;
        std::set<Cell> uncached_free;
        if (PyCapsule_IsValid(free_object,GRID_NAME)) {
            grid=static_cast<NativeGrid*>(PyCapsule_GetPointer(free_object,GRID_NAME));
        } else {
            auto free_vector=cells(free_object);
            uncached_free.insert(free_vector.begin(),free_vector.end());
        }
        const auto& free=grid ? grid->free : uncached_free;
        auto blocked_vector = cells(blocked_object);
        std::set<Cell> blocked(blocked_vector.begin(), blocked_vector.end());
        Cell start = cell(start_object); blocked.erase(start);
        std::set<Cell> goals;
        for (auto p: cells(goal_object)) if (free.count(p) && !blocked.count(p)) goals.insert(p);
        std::map<Cell, Cell> parent;
        std::vector<Cell> endpoints, settled;
        if (free.count(start) && !goals.empty() && limit > 0) {
            using Entry = std::tuple<double, long, long>;
            std::priority_queue<Entry, std::vector<Entry>, std::greater<Entry>> queue;
            std::map<Cell, double> costs{{start, 0.}};
            queue.emplace(0., start.first, start.second);
            const long moves[][2]={{-1,0},{1,0},{0,-1},{0,1},{-1,-1},{-1,1},{1,-1},{1,1}};
            while (!queue.empty() && !goals.empty() && endpoints.size() < limit) {
                auto [cost, x, y] = queue.top(); queue.pop(); Cell current{x,y};
                if (cost != costs.at(current)) continue;
                settled.push_back(current);
                if (goals.erase(current)) endpoints.push_back(current);
                auto relax=[&](Cell next,bool diagonal) {
                    double next_cost=cost + (diagonal ? std::sqrt(2.) : 1.);
                    auto previous=costs.find(next);
                    if (previous != costs.end() && next_cost >= previous->second) return;
                    costs[next]=next_cost; parent[next]=current;
                    queue.emplace(next_cost,next.first,next.second);
                };
                if (grid) {
                    for (const auto& edge:grid->neighbors.at(current)) {
                        if (blocked.count(edge.next) || (edge.diagonal &&
                            (blocked.count(edge.side1) || blocked.count(edge.side2)))) continue;
                        relax(edge.next,edge.diagonal);
                    }
                } else for (auto& move: moves) {
                    long dx=move[0], dy=move[1]; Cell next{x+dx,y+dy};
                    if (!free.count(next) || blocked.count(next)) continue;
                    if (dx && dy && (!free.count({x+dx,y}) || blocked.count({x+dx,y}) ||
                                     !free.count({x,y+dy}) || blocked.count({x,y+dy}))) continue;
                    relax(next,dx && dy);
                }
            }
        }
        Ref parents(PyDict_New()); checked(parents.p);
        for (auto [a,b]:parent) {
            Ref key(point(a)), value(point(b)); checked(key.p && value.p);
            checked(PyDict_SetItem(parents.p,key.p,value.p)==0);
        }
        Ref ends(points(endpoints)), visits(points(settled));
        return PyTuple_Pack(3, parents.p, ends.p, visits.p);
    } catch (...) { return failure(); }
}

static PyObject* prefixes(PyObject*, PyObject* args) {
    PyObject *parent_object,*settled_object,*start_object,*forbidden_object,*centres_object;
    PyObject* cache_object=Py_None;
    double clearance,grid;
    if (!PyArg_ParseTuple(args,"OOOOOdd|O",&parent_object,&settled_object,&start_object,
                         &forbidden_object,&centres_object,&clearance,&grid,&cache_object)) return nullptr;
    try {
        checked(PyDict_Check(parent_object));
        std::map<Cell,Cell> parent;
        Py_ssize_t index=0; PyObject *key,*value;
        while (PyDict_Next(parent_object,&index,&key,&value)) parent[cell(key)]=cell(value);
        auto visits=cells(settled_object), blocked=cells(forbidden_object);
        std::set<Cell> forbidden(blocked.begin(),blocked.end());
        Cell start=cell(start_object);
        Ref centres_seq(PySequence_Fast(centres_object,"expected centres")); checked(centres_seq.p);
        checked(PySequence_Fast_GET_SIZE(centres_seq.p) <= 63);
        std::vector<std::pair<double,double>> centres;
        for (Py_ssize_t i=0;i<PySequence_Fast_GET_SIZE(centres_seq.p);++i) {
            Ref centre(PySequence_Fast(PySequence_Fast_GET_ITEM(centres_seq.p,i),"expected centre"));
            checked(centre.p && PySequence_Fast_GET_SIZE(centre.p)==2);
            double x=PyFloat_AsDouble(PySequence_Fast_GET_ITEM(centre.p,0));
            double y=PyFloat_AsDouble(PySequence_Fast_GET_ITEM(centre.p,1));
            checked(!PyErr_Occurred()); centres.emplace_back(x,y);
        }
        auto separation=[](Cell p, std::pair<double,double> c) {
            return std::hypot(double(p.first)-c.first,double(p.second)-c.second);
        };
        BodyCache* cache=nullptr;
        std::map<EdgeKey,std::pair<bool,uint64_t>>* transitions=nullptr;
        if (cache_object!=Py_None) {
            cache=static_cast<BodyCache*>(PyCapsule_GetPointer(cache_object,CACHE_NAME));
            checked(cache);
            BodyKey key{centres,clearance};
            auto found=cache->groups.find(key);
            if (found!=cache->groups.end()) transitions=&found->second;
            else if (cache->groups.size()<64) transitions=&cache->groups[key];
        }
        struct Prefix { bool valid,exited; uint64_t mask; double distance; };
        uint64_t initial=0;
        for (size_t i=0;i<centres.size();++i) if (separation(start,centres[i])>clearance) initial |= uint64_t(1)<<i;
        std::map<Cell,Prefix> result{{start,{true,!forbidden.count(start),initial,0.}}};
        for (Cell p:visits) {
            if (p==start) continue;
            Cell previous=parent.at(p); Prefix state=result.at(previous);
            state.distance += std::hypot(double(p.first-previous.first),double(p.second-previous.second))*grid;
            if (forbidden.count(p)) state.valid=state.valid && !state.exited;
            else state.exited=true;
            if (state.valid) {
              EdgeKey edge{previous.first,previous.second,p.first,p.second,state.mask};
              bool hit=false;
              if (transitions) {
                  auto found=transitions->find(edge);
                  if (found!=transitions->end()) {
                      hit=true; ++cache->hits;
                      state.valid=found->second.first; state.mask=found->second.second;
                  }
              }
              if (!hit) {
               if (cache) ++cache->misses;
               for (size_t i=0;i<centres.size();++i) {
                double d=separation(p,centres[i]); uint64_t bit=uint64_t(1)<<i;
                if (state.mask & bit) {
                    if (d<=clearance+1e-9) {state.valid=false;break;}
                } else {
                    if (d+1e-9<separation(previous,centres[i])) {state.valid=false;break;}
                    if (d>clearance+1e-9) state.mask |= bit;
                }
               }
               if (transitions && cache->entries<32768) {
                   (*transitions)[edge]={state.valid,state.mask}; ++cache->entries;
               }
              }
            }
            result[p]=state;
        }
        Ref output(PyDict_New()); checked(output.p);
        for (auto [p,s]:result) {
            Ref k(point(p)), v(Py_BuildValue("(OOKd)",s.valid ? Py_True:Py_False,
                                           s.exited ? Py_True:Py_False,
                                           static_cast<unsigned long long>(s.mask),s.distance));
            checked(k.p && v.p && PyDict_SetItem(output.p,k.p,v.p)==0);
        }
        return Py_NewRef(output.p);
    } catch (...) {return failure();}
}

static PyObject* astar_bodies(PyObject*,PyObject* args) {
    PyObject *free_object,*start_object,*goal_object,*centres_object,*blocked_object;
    double clearance;
    if (!PyArg_ParseTuple(args,"OOOOdO",&free_object,&start_object,&goal_object,
                         &centres_object,&clearance,&blocked_object)) return nullptr;
    try {
        auto free_vector=cells(free_object), blocked_vector=cells(blocked_object);
        std::set<Cell> free(free_vector.begin(),free_vector.end());
        std::set<Cell> blocked(blocked_vector.begin(),blocked_vector.end());
        Cell start=cell(start_object), goal=cell(goal_object); blocked.erase(start);
        if (!free.count(start) || !free.count(goal) || blocked.count(goal) || clearance<=0) Py_RETURN_NONE;
        Ref seq(PySequence_Fast(centres_object,"expected centres")); checked(seq.p);
        checked(PySequence_Fast_GET_SIZE(seq.p)<=63);
        std::vector<std::pair<double,double>> centres;
        for (Py_ssize_t i=0;i<PySequence_Fast_GET_SIZE(seq.p);++i) {
            Ref c(PySequence_Fast(PySequence_Fast_GET_ITEM(seq.p,i),"expected centre"));
            checked(c.p && PySequence_Fast_GET_SIZE(c.p)==2);
            double x=PyFloat_AsDouble(PySequence_Fast_GET_ITEM(c.p,0));
            double y=PyFloat_AsDouble(PySequence_Fast_GET_ITEM(c.p,1));
            checked(!PyErr_Occurred()); centres.emplace_back(x,y);
        }
        auto distance=[](Cell p,std::pair<double,double> c) {
            return std::hypot(double(p.first)-c.first,double(p.second)-c.second);
        };
        auto heuristic=[goal](Cell p) {
            double dx=std::abs(double(goal.first-p.first)), dy=std::abs(double(goal.second-p.second));
            return std::max(dx,dy)+(std::sqrt(2.)-1.)*std::min(dx,dy);
        };
        uint64_t initial_mask=0, all_cleared=(uint64_t(1)<<centres.size())-1;
        for (size_t i=0;i<centres.size();++i)
            if (distance(start,centres[i])>clearance) initial_mask |= uint64_t(1)<<i;
        using State=std::tuple<long,long,uint64_t>;
        using Entry=std::tuple<double,double,long,long,uint64_t>;
        State initial{start.first,start.second,initial_mask};
        std::priority_queue<Entry,std::vector<Entry>,std::greater<Entry>> queue;
        queue.emplace(heuristic(start),0.,start.first,start.second,initial_mask);
        std::map<State,double> costs{{initial,0.}};
        std::map<State,State> parent;
        const long moves[][2]={{-1,0},{1,0},{0,-1},{0,1},{-1,-1},{-1,1},{1,-1},{1,1}};
        while (!queue.empty()) {
            auto [f,cost,x,y,mask]=queue.top(); queue.pop();
            State state{x,y,mask}; Cell current{x,y};
            if (cost!=costs.at(state)) continue;
            if (current==goal && mask==all_cleared) {
                std::vector<Cell> path{current};
                while (parent.count(state)) {
                    state=parent.at(state);
                    path.emplace_back(std::get<0>(state),std::get<1>(state));
                }
                std::reverse(path.begin(),path.end());
                return points(path);
            }
            for (auto& move:moves) {
                long dx=move[0],dy=move[1]; Cell next{x+dx,y+dy};
                if (!free.count(next) || blocked.count(next)) continue;
                if (dx && dy && (!free.count({x+dx,y}) || blocked.count({x+dx,y}) ||
                                 !free.count({x,y+dy}) || blocked.count({x,y+dy}))) continue;
                uint64_t next_mask=mask; bool valid=true;
                for (size_t i=0;i<centres.size();++i) {
                    uint64_t bit=uint64_t(1)<<i;
                    double next_distance=distance(next,centres[i]);
                    if (mask & bit) {
                        if (next_distance<=clearance+1e-9) {valid=false;break;}
                        continue;
                    }
                    // Unlike clearance prefix validation, continuation A*
                    // requires STRICT outward progress while still touching.
                    if (next_distance<=distance(current,centres[i])+1e-9) {valid=false;break;}
                    if (next_distance>clearance+1e-9) next_mask |= bit;
                }
                if (!valid) continue;
                State next_state{next.first,next.second,next_mask};
                double next_cost=cost+((dx && dy) ? std::sqrt(2.):1.);
                auto old=costs.find(next_state);
                if (old!=costs.end() && next_cost>=old->second) continue;
                costs[next_state]=next_cost; parent[next_state]=state;
                queue.emplace(next_cost+heuristic(next),next_cost,next.first,next.second,next_mask);
            }
        }
        Py_RETURN_NONE;
    } catch (...) {return failure();}
}

// Batch the segment loop, retaining CPython's hypot rounding (std::hypot can
// differ by an ulp and change clearance tie-breaks).
static PyObject* endpoint_margins(PyObject*, PyObject* args) {
    PyObject *endpoints_object, *prepared_object;
    double radius, grid_size;
    if (!PyArg_ParseTuple(args,"OOdd",&endpoints_object,&prepared_object,
                         &radius,&grid_size)) return nullptr;
    try {
        auto endpoints=cells(endpoints_object);
        Ref routes(PySequence_Fast(prepared_object,"expected prepared routes"));
        checked(routes.p);
        using Segment=std::tuple<double,double,double,double,double>;
        struct Route {std::vector<Segment> segments; size_t vertices; double x=0,y=0;};
        std::vector<Route> geometry;
        for (Py_ssize_t i=0;i<PySequence_Fast_GET_SIZE(routes.p);++i) {
            Ref route(PySequence_Fast(PySequence_Fast_GET_ITEM(routes.p,i),"expected route"));
            checked(route.p && PySequence_Fast_GET_SIZE(route.p)==2);
            Ref vertices(PySequence_Fast(PySequence_Fast_GET_ITEM(route.p,0),"expected vertices"));
            Ref segments(PySequence_Fast(PySequence_Fast_GET_ITEM(route.p,1),"expected segments"));
            checked(vertices.p && segments.p);
            Route value; value.vertices=PySequence_Fast_GET_SIZE(vertices.p);
            if (value.vertices==1) {
                Ref vertex(PySequence_Fast(PySequence_Fast_GET_ITEM(vertices.p,0),"expected vertex"));
                checked(vertex.p && PySequence_Fast_GET_SIZE(vertex.p)==2);
                value.x=PyFloat_AsDouble(PySequence_Fast_GET_ITEM(vertex.p,0));
                value.y=PyFloat_AsDouble(PySequence_Fast_GET_ITEM(vertex.p,1));
            }
            for (Py_ssize_t j=0;j<PySequence_Fast_GET_SIZE(segments.p);++j) {
                Ref segment(PySequence_Fast(PySequence_Fast_GET_ITEM(segments.p,j),"expected segment"));
                checked(segment.p && PySequence_Fast_GET_SIZE(segment.p)==5);
                double v[5];
                for (int k=0;k<5;++k) v[k]=PyFloat_AsDouble(PySequence_Fast_GET_ITEM(segment.p,k));
                value.segments.emplace_back(v[0],v[1],v[2],v[3],v[4]);
            }
            checked(!PyErr_Occurred()); geometry.push_back(std::move(value));
        }
        Ref math(PyImport_ImportModule("math")); checked(math.p);
        Ref hypot(PyObject_GetAttrString(math.p,"hypot")); checked(hypot.p);
        auto norm=[&](double x,double y) {
            Ref a(PyFloat_FromDouble(x)), b(PyFloat_FromDouble(y)); checked(a.p && b.p);
            PyObject* values[]={a.p,b.p};
            Ref result(PyObject_Vectorcall(hypot.p,values,2,nullptr)); checked(result.p);
            return PyFloat_AsDouble(result.p);
        };
        Ref result(PyList_New(endpoints.size())); checked(result.p);
        for (size_t i=0;i<endpoints.size();++i) {
            double px=endpoints[i].first,py=endpoints[i].second;
            double margin=INFINITY;
            for (const auto& route:geometry) {
                double best=INFINITY;
                if (route.vertices==1) best=norm(px-route.x,py-route.y);
                for (const auto& segment:route.segments) {
                    auto [ax,ay,dx,dy,squared]=segment;
                    double ratio=squared<=1e-12 ? 0. : ((px-ax)*dx+(py-ay)*dy)/squared;
                    ratio=std::max(0.,std::min(1.,ratio));
                    best=std::min(best,norm(px-(ax+ratio*dx),py-(ay+ratio*dy)));
                }
                margin=std::min(margin,best);
            }
            PyObject* value=PyFloat_FromDouble(margin*grid_size-2*radius); checked(value);
            PyList_SET_ITEM(result.p,i,value);
        }
        return Py_NewRef(result.p);
    } catch (...) {return failure();}
}

static PyMethodDef methods[]={
    {"prepare_grid",prepare_grid,METH_O,"Immutable static adjacency in original move order."},
    {"endpoint_margins",endpoint_margins,METH_VARARGS,"Batch exact CPython-rounded clearance margins."},
    {"astar_bodies",astar_bodies,METH_VARARGS,"Continuation A* with strict contact separation."},
    {"new_clearance_cache",new_cache,METH_NOARGS,"Create a bounded snapshot-local edge cache."},
    {"clearance_cache_stats",cache_stats,METH_O,"Read cache reuse counters."},
    {"shortest_path_tree",tree,METH_VARARGS,"Bulk Dijkstra preserving Python heap ties."},
    {"clearance_prefixes",prefixes,METH_VARARGS,"Bulk prefix geometry validation."},
    {nullptr,nullptr,0,nullptr}
};
static PyModuleDef module={PyModuleDef_HEAD_INIT,"_grid_geometry_native",nullptr,-1,methods};
PyMODINIT_FUNC PyInit__grid_geometry_native() {
    PyObject* m=PyModule_Create(&module);
    if (m) PyModule_AddStringConstant(m,"revision","grid-geometry-v5");
    return m;
}
