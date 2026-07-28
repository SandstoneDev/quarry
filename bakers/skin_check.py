#!/usr/bin/env python3
"""Find the exact bind-pose skinning convention on real fam1 data using an affine
(R 3x3 + t) representation so the RW pad-column never corrupts the math. For the
bind pose skinMat[b] = invBind[b] o bindWorld[b] must be IDENTITY. Brute-force the
remaining ambiguities (frame-rot transpose, invBind-rot transpose, compose order,
skin compose order) and report the identity combo."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sa_skin


def t3(R):  # transpose 3x3 (row-major 9)
    return [R[0],R[3],R[6], R[1],R[4],R[7], R[2],R[5],R[8]]

def rmul(A, B):  # 3x3 * 3x3
    C=[0.0]*9
    for r in range(3):
        for c in range(3):
            C[r*3+c]=sum(A[r*3+k]*B[k*3+c] for k in range(3))
    return C

def vmulR(v, R):  # row vector v(3) * R(3x3)
    return [v[0]*R[0]+v[1]*R[3]+v[2]*R[6],
            v[0]*R[1]+v[1]*R[4]+v[2]*R[7],
            v[0]*R[2]+v[1]*R[5]+v[2]*R[8]]

def compose(a, b):  # apply a then b (row-vec): p*a*b
    Ra,ta=a; Rb,tb=b
    return (rmul(Ra,Rb), [vmulR(ta,Rb)[i]+tb[i] for i in range(3)])

I3=[1,0,0,0,1,0,0,0,1]
def err(M):
    R,t=M
    re=max(abs(R[i]-I3[i]) for i in range(9))
    te=max(abs(x) for x in t)
    return re,te


def main():
    ped = sys.argv[1] if len(sys.argv)>1 else "fam1"
    im=sa_skin.sa_img.SaImg(sa_skin.GTA3); sk=sa_skin.decode(im.extract(ped+".dff"))
    frames,nodes,geo=sk["frames"],sk["nodes"],sk["geoms"][0]
    nb=geo["numBones"]
    fbn={f["nodeId"]:i for i,f in enumerate(frames) if f["nodeId"]>=0}
    # invBind -> (R3x3 from m[0..2,4..6,8..10], t from m[12..14])
    IB=[]
    for m in geo["invBind"]:
        IB.append(([m[0],m[1],m[2], m[4],m[5],m[6], m[8],m[9],m[10]], [m[12],m[13],m[14]]))

    def world(order, trot):
        W=[None]*len(frames)
        def loc(i):
            R=frames[i]["rot"]
            if trot: R=t3(R)
            return (list(R), list(frames[i]["pos"]))
        def rec(i):
            if W[i] is not None: return W[i]
            p=frames[i]["parent"]
            if p is None or p<0: W[i]=loc(i)
            else:
                pw=rec(p)
                W[i]=compose(loc(i),pw) if order=='lp' else compose(pw,loc(i))
            return W[i]
        for i in range(len(frames)): rec(i)
        return W

    best=[]
    for order in ('lp','pl'):
        for trot in (False,True):
            W=world(order,trot)
            for tirot in (False,True):
                for so in ('iw','wi'):  # invBind o world / world o invBind
                    re_t=te_t=0.0; ok=0
                    for b in range(nb):
                        fi=fbn.get(nodes[b][0],-1)
                        if fi<0: re_t=9e9; break
                        Rib,tib=IB[b]
                        if tirot: Rib=t3(Rib)
                        ib=(Rib,tib)
                        sm=compose(ib,W[fi]) if so=='iw' else compose(W[fi],ib)
                        re,te=err(sm); re_t+=re; te_t+=te
                        if re<1e-3 and te<1e-2: ok+=1
                    best.append((re_t/nb,te_t/nb,ok,order,trot,tirot,so))
    best.sort(key=lambda x:x[0]+x[1])
    print("rotErr transErr #ident comp ftrot ibrot skinOrder")
    for (re,te,ok,order,trot,tirot,so) in best[:8]:
        print("  %.4f %.4f %3d/%d  %s ftrot=%-5s ibrot=%-5s %s %s"
              %(re,te,ok,nb,order,trot,tirot,so,"<== IDENTITY" if ok==nb else ""))


if __name__=="__main__":
    main()
