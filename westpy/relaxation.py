import glob
import json
import numpy as np
import os
import re
import shutil
import subprocess
import xml.etree.ElementTree as ET
import yaml
from westpy.units import Angstrom, Hartree, Rydberg


class bfgs_iter:
    """Class for carrying out BFGS geometry relaxation.

    :Example:

    >>> from westpy import *
    >>> run_pw = "mpirun -n 1 pw.x"
    >>> run_wbse = "mpirun -n 4 wbse.x -nb 4"
    >>> run_wbse_init = "mpirun -n 4 wbse_init.x -ni 4"
    >>> bfgs = bfgs_iter(run_pw=run_pw, run_wbse=run_wbse, grad_thr=1e-4, maxiter=30)
    >>> bfgs.solve()

    """

    def __init__(
        self,
        run_pw: str,
        run_wbse: str,
        run_wbse_init: str = None,
        pp: list = [],
        pw_input: str = "pw.in",
        wbse_input: str = "wbse.in",
        wbse_init_input: str = None,
        l_restart: bool = False,
        energy_thr: float = 1.0e-4,
        grad_thr: float = 1.0e-3,
        maxiter: int = 100,
        w1: float = 0.01,
        w2: float = 0.5,
        bfgs_ndim: int = 1,
        trust_radius_ini: float = 0.5,
        trust_radius_min: float = 2.0e-4,
        trust_radius_max: float = 0.8,
    ):
        """
        run_pw: Full command to run pwscf, e.g., mpirun -n 2 /path/to/qe/bin/pw.x
        run_wbse: Full command to run wbse, e.g., mpirun -n 4 /path/to/qe/bin/wbse.x -nb 4
        run_wbse_init: Full command to run wbse_init, e.g., mpirun -n 4 /path/to/qe/bin/wbse_init.x
        pp: List of pseudopotential files
        pw_input: pw.x input file name
        wbse_input: wbse.x input file name
        wbse_init_input: wbse_init.x input file name (default: same as wbse_input)
        l_restart: If True, restart an unfinished run
        energy_thr: Convergence threshold on total energy (Ry) for ionic minimization
        grad_thr: Convergence threshold on forces (Ry/Bohr) for ionic minimization
        maxiter: Maximum number of BFGS steps
        w1: Parameters used in line search based on the Wolfe conditions
        w2: Parameters used in line search based on the Wolfe conditions
        bfgs_ndim: Dimension of BFGS. Only bfgs_ndim == 1 implemented
        trust_radius_ini: Initial ionic displacement in the structural relaxation
        trust_radius_min: Minimum ionic displacement in the structural relaxation
        trust_radius_max: Maximum ionic displacement in the structural relaxation

        """
        # how to run
        self.run_pw = run_pw
        self.run_wbse = run_wbse
        self.run_wbse_init = run_wbse_init
        self.pp = pp
        self.pw_input = pw_input
        if wbse_init_input:
            self.wbse_init_input = wbse_init_input
        else:
            self.wbse_init_input = wbse_input
        self.wbse_input = wbse_input

        # how to do BFGS
        self.energy_thr = energy_thr
        self.grad_thr = grad_thr
        self.maxiter = maxiter
        self.w1 = w1
        self.w2 = w2
        self.bfgs_ndim = bfgs_ndim
        assert self.bfgs_ndim == 1, "bfgs_ndim > 1 not implemented"
        self.trust_radius_ini = trust_radius_ini
        self.trust_radius_min = trust_radius_min
        self.trust_radius_max = trust_radius_max

        # internal parameters
        self.folder_name = "step-"
        self.tmp_file = "bfgs_tmp.json"
        self.l_exx = False
        self.conv_bfgs = False
        if l_restart:
            self.start_iter = self.read_restart()
        else:
            self.start_iter = 0
        self.read_prefix()
        self.read_pos_unit()

    def log(self, string: str, indent: int = 5):
        """
        write bfgs information
        string: a string of message
        indent: indentation level
        """
        print(" " * indent + string, flush=True)

    def read_restart(self):
        bfgs_tmp_file = os.getcwd() + "/" + self.tmp_file
        with open(bfgs_tmp_file) as f:
            bfgs_data = json.load(f)
        return int(bfgs_data["scf_iter"])

    def read_prefix(self):
        wbse_in = os.getcwd() + "/" + self.wbse_input

        with open(wbse_in, "r") as f:
            data = yaml.load(f, Loader=yaml.SafeLoader)

        self.pw_prefix = "pwscf"
        self.west_prefix = "west"
        self.outdir = "./"

        if "input_west" in data:
            if "qe_prefix" in data["input_west"]:
                self.pw_prefix = data["input_west"]["qe_prefix"]
            if "west_prefix" in data["input_west"]:
                self.west_prefix = data["input_west"]["west_prefix"]
            if "outdir" in data["input_west"]:
                self.outdir = data["input_west"]["outdir"] + "/"

    def read_pos_unit(self):
        pw_in = os.getcwd() + "/" + self.pw_input

        # extract the unit for ATOMIC_POSITIONS from pw.in
        with open(pw_in, "r") as f:
            for line in f:
                match = re.search(
                    r"^ATOMIC_POSITIONS[ \t]+(?:\(|\{)?(\w+)(?:\)|\})?", line
                )
                if match:
                    pos_unit = match.group(1).lower()
                    break

        if not pos_unit in ["angstrom", "crystal", "bohr"]:
            self.log(f"{pos_unit} unit for ATOMIC_POSITIONS is currently not supported")
            exit()

        self.pos_unit = pos_unit

    def run_calc(self):
        root_dir = os.getcwd() + "/"
        work_dir = root_dir + self.folder_name + str(self.scf_iter) + "/"

        if self.scf_iter == 1:
            os.mkdir(work_dir)
            shutil.copy2(self.pw_input, work_dir + self.pw_input)

        # run pw
        self.log("Running pw.x ...")
        for pp in self.pp:
            shutil.copy2(pp, work_dir + pp)
        command = f"{self.run_pw} -i {self.pw_input} > pw.out"
        try:
            subprocess.run(command, shell=True, cwd=work_dir, check=True)
        except subprocess.CalledProcessError:
            self.log(f"pw.x failed: {work_dir}")
            exit()

        # load number of atoms, lattice parameters, atomic species, and atomic positions
        self.read_constants_positions_pwxml()
        # load ground state forces and energy
        self.read_ground_state_forces_energy_pwxml()

        # run wbse_init
        if self.l_exx:
            if self.run_wbse_init is None:
                self.log("EXX calculations require wbse_init")
                exit()

            self.log("Running wbse_init.x ...")
            shutil.copy2(self.wbse_init_input, work_dir + self.wbse_init_input)
            command = f"{self.run_wbse_init} -i {self.wbse_init_input} > wbse_init.out"
            try:
                subprocess.run(command, shell=True, cwd=work_dir, check=True)
            except subprocess.CalledProcessError:
                self.log(f"wbse_init.x failed: {work_dir}")
                exit()

        # run wbse
        self.log("Running wbse.x ...")
        json_file = work_dir + self.outdir + self.west_prefix + ".wbse.save/wbse.json"
        if os.path.exists(json_file):
            os.remove(json_file)
        shutil.copy2(self.wbse_input, work_dir + self.wbse_input)
        command = f"{self.run_wbse} -i {self.wbse_input} > wbse.out"
        try:
            subprocess.run(command, shell=True, cwd=work_dir, check=True)
        except subprocess.CalledProcessError:
            self.log(f"wbse.x failed: {work_dir}")
            exit()

        # load excited state forces and energy
        self.read_excited_state_forces_energy()

    def initialize(self):
        self.lwolfe = False

        # lattice parameters in Bohr
        h = self.alat * self.at
        hinv = np.linalg.inv(h)
        self.hinv_block = np.zeros((3 * self.nat, 3 * self.nat))
        for k in range(self.nat):
            for i in range(3):
                for j in range(3):
                    self.hinv_block[i + 3 * k, j + 3 * k] = hinv[i, j]

        g = np.dot(h.T, h)
        self.metric = np.zeros((3 * self.nat, 3 * self.nat))
        for k in range(self.nat):
            for i in range(3):
                for j in range(3):
                    self.metric[i + 3 * k, j + 3 * k] = g[i, j]

        self.pos = self.pos_in
        self.grad = self.grad_in

        if self.scf_iter == 1:
            self.inv_hess = np.linalg.inv(self.metric)
            self.pos_p = np.zeros(self.nat * 3)
            self.grad_p = np.zeros(self.nat * 3)
            self.bfgs_iter = 0
            self.gdiis_iter = 0
            self.energy_p = self.energy
            self.step_old = np.zeros(self.nat * 3)
            self.nr_step_length = 0.0
            self.trust_radius_old = self.trust_radius_ini
            self.pos_old = np.zeros(self.nat * 3)
            self.grad_old = np.zeros(self.nat * 3)
            self.tr_min_hit = 0
        else:
            # load from bfgs.tmp
            bfgs_tmp_file = os.getcwd() + "/" + self.tmp_file
            with open(bfgs_tmp_file) as f:
                bfgs_data = json.load(f)

            self.pos_p = np.array(bfgs_data["pos"])
            self.grad_p = np.array(bfgs_data["grad"])
            if self.scf_iter != int(bfgs_data["scf_iter"]) + 1:
                self.log("unexpected scf_iter error")
                exit()
            self.bfgs_iter = int(bfgs_data["bfgs_iter"])
            self.gdiis_iter = int(bfgs_data["gdiis_iter"])
            self.energy_p = float(bfgs_data["energy"])
            self.pos_old = np.array(bfgs_data["pos_old"])
            self.grad_old = np.array(bfgs_data["grad_old"])
            self.inv_hess = np.array(bfgs_data["inv_hess"])
            self.tr_min_hit = int(bfgs_data["tr_min_hit"])
            self.nr_step_length = float(bfgs_data["nr_step_length"])

            self.step_old = self.pos - self.pos_p
            self.trust_radius_old = self.scnorm(self.step_old)
            self.step_old = self.step_old / self.trust_radius_old

    def solve(self):
        self.log("BFGS Geometry Relaxation")
        self.log("")

        for scf_iter in range(self.start_iter, self.maxiter):
            self.scf_iter = scf_iter + 1

            self.run_calc()
            self.initialize()

            self.log("")
            self.log(f"number of scf cycles    = {self.scf_iter:3d}")
            self.log(f"number of bfgs steps    = {self.bfgs_iter:3d}")
            if self.scf_iter > 1:
                self.log(f"energy old              = {self.energy_p:18.10f} Ry")
            self.log(f"energy new              = {self.energy:18.10f} Ry")
            self.grad_t = np.linalg.norm(np.dot(self.hinv_block.T, self.grad))
            self.log(f"total force             = {self.grad_t:18.10f} Ry/Bohr")

            energy_error = abs(self.energy_p - self.energy)
            grad_error = np.max(abs(np.dot(self.hinv_block.T, self.grad)))
            self.conv_bfgs = energy_error < self.energy_thr
            self.conv_bfgs = self.conv_bfgs and (grad_error < self.grad_thr)
            self.conv_bfgs = self.conv_bfgs or (self.tr_min_hit > 1)
            if self.conv_bfgs:
                self.pos_in = self.pos
                self.grad_in = self.grad
                self.save_final_geo()
                break

            self.compute_new_position()
            self.write_new_pw_input()
            self.log("")

        self.terminate_bfgs()

    def save_final_geo(self):
        # save final geometry
        root_dir = os.getcwd() + "/"
        save_dir = root_dir + "final_geo"
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)
        last_dir = root_dir + self.folder_name + str(self.scf_iter)
        shutil.copytree(last_dir, save_dir)
        self.log("")
        self.log(f"Final geometry saved to {save_dir}")

        # clean up
        bfgs_tmp_file = root_dir + self.tmp_file
        if os.path.exists(bfgs_tmp_file):
            os.remove(bfgs_tmp_file)

        for folder in glob.glob(f"{root_dir}{self.folder_name}*"):
            shutil.rmtree(folder)

    def compute_new_position(self):
        # energy wolfe condition
        self.check_energy_condition()
        if (not self.e_wolfe) and (self.scf_iter > 1):
            # the previous step is rejected, line search goes on
            self.step_accepted = False
            #
            self.log("CASE: energy_new > energy_old")

            if abs(self.scnorm(self.step_old) - 1.0) > 1.0e-10:
                self.log("step_old is NOT normalized")
                exit()

            self.step = self.step_old
            self.dE0s = np.dot(self.grad_p, self.step) * self.trust_radius_old

            if self.dE0s > 0.0:
                self.log("dE0s is positive which should never happen")
                exit()

            self.den = self.energy - self.energy_p - self.dE0s
            self.trust_radius = -0.5 * self.dE0s * self.trust_radius_old / self.den
            self.log(f"new trust radius        = {self.trust_radius:18.10f} Bohr")
            self.pos = self.pos_p
            self.energy = self.energy_p
            self.grad = self.grad_p
            if self.trust_radius < self.trust_radius_min:
                # the history is reset
                self.log("trust_radius < trust_radius_min")
                self.log("resetting bfgs history")

                if self.tr_min_hit == 1:
                    # something is going wrong
                    self.log("history already reset at previous step: stopping")
                    self.tr_min_hit = 2
                else:
                    self.tr_min_hit = 1

                self.reset_bfgs()
                self.step = -np.dot(self.inv_hess, self.grad)
                self.nr_step_length = self.scnorm(self.step)
                self.step = self.step / self.nr_step_length
                self.trust_radius = min(self.trust_radius_ini, self.nr_step_length)
                #
            else:
                self.tr_min_hit = 0
        else:
            # a new bfgs step is done
            self.bfgs_iter = self.bfgs_iter + 1

            if self.bfgs_iter == 1:
                self.step_accepted = False
            else:
                self.step_accepted = True
                self.nr_step_length_old = self.nr_step_length
                self.log("CASE: energy_new < energy_old")
                self.check_wolfe_conditions()
                self.update_inverse_hessian()

            self.step = -np.dot(self.inv_hess, self.grad)

            if np.dot(self.grad, self.step) > 0.0:
                # resetting bfgs
                self.log("resetting bfgs history")
                self.inv_hess = self.reset_bfgs()
                self.step = -np.dot(self.inv_hess, self.grad)

            self.nr_step_length = self.scnorm(self.step)
            self.step = self.step / self.nr_step_length

            # compute the new trust radius
            if self.bfgs_iter == 1:
                self.trust_radius = min(self.trust_radius_ini, self.nr_step_length)
                self.tr_min_hit = 0
            else:
                self.compute_trust_radius()

            self.log(f"new trust radius        = {self.trust_radius:18.10f} Bohr")

        if self.nr_step_length < 1.0e-16:
            self.log("NR step-length unreasonably short")
            exit()

        # information required by next iteration is saved here
        # (this must be done before positions are updated)
        bfgs_data = {
            "pos": self.pos.tolist(),
            "grad": self.grad.tolist(),
            "scf_iter": self.scf_iter,
            "bfgs_iter": self.bfgs_iter,
            "gdiis_iter": self.gdiis_iter,
            "energy": self.energy,
            "pos_old": self.pos_old.tolist(),
            "grad_old": self.grad_old.tolist(),
            "inv_hess": self.inv_hess.tolist(),
            "tr_min_hit": self.tr_min_hit,
            "nr_step_length": self.nr_step_length,
        }
        bfgs_tmp_file = os.getcwd() + "/" + self.tmp_file
        with open(bfgs_tmp_file, "w") as f:
            json.dump(bfgs_data, f, indent=4)

        # new position
        self.pos = self.pos + self.trust_radius * self.step

    def write_new_pw_input(self):
        # convert unit of atomic positions if necessary
        if self.pos_unit == "bohr":
            pos_to_write = np.dot(self.pos, self.h)
        elif self.pos_unit == "angstrom":
            pos_to_write = np.dot(self.pos, self.h) / Angstrom
        elif self.pos_unit == "crystal":
            pos_to_write = self.pos

        root_dir = os.getcwd() + "/"
        next_dir = root_dir + self.folder_name + str(self.scf_iter + 1) + "/"
        if os.path.exists(next_dir):
            shutil.rmtree(next_dir)
        os.mkdir(next_dir)

        old_pw_in = root_dir + self.pw_input
        new_pw_in = next_dir + self.pw_input

        with open(old_pw_in, "r") as f:
            lines = f.readlines()

        for start, line in enumerate(lines):
            if line:
                if line.split()[0] == "ATOMIC_POSITIONS":
                    break

        # update atomic positions
        for iat in range(self.nat):
            new_line = f"{self.atoms[iat]}"
            for iforce in range(3):
                new_line += f"   {pos_to_write[3 * iat + iforce]:.12f}"
            new_line += "\n"

            lines[start + 1 + iat] = new_line

        with open(new_pw_in, "w") as f:
            f.writelines(lines)

        work_dir = root_dir + self.folder_name + str(self.scf_iter) + "/"
        shutil.rmtree(work_dir)

    def reset_bfgs(self):
        """
        inv_hess is re-initialized to initial guess
        """
        self.inv_hess = np.linalg.inv(self.metric)

    def update_inverse_hessian(self):
        """
        update inverse hessian
        """
        s = self.pos - self.pos_p
        y = self.grad - self.grad_p
        sdoty = np.dot(s, y)

        if abs(sdoty) < 1e-16:
            self.reset_bfgs()
        else:
            yH = s
            yH = np.dot(np.linalg.inv(self.inv_hess), yH)
            sBs = np.dot(s, yH)
            if sdoty < 0.2 * sBs:
                Theta = 0.8 * sBs / (sBs - sdoty)
                y = Theta * y + (1.0 - Theta) * yH

        Hy = np.dot(self.inv_hess, y)
        yH = np.dot(y, self.inv_hess)
        self.inv_hess = self.inv_hess + 1.0 / sdoty * (
            (1.0 + np.dot(y, Hy) / sdoty) * np.outer(s, s)
            - (np.outer(s, yH) + np.outer(Hy, s))
        )

    def check_wolfe_conditions(self):
        """
        check wolfe conditions
        """
        g_wolfe = abs(np.dot(self.grad, self.step_old)) < -self.w2 * np.dot(
            self.grad_p, self.step_old
        )
        self.lwolfe = self.e_wolfe and g_wolfe

    def check_energy_condition(self):
        """
        check energy condition
        """
        self.e_wolfe = (self.energy - self.energy_p) < self.w1 * np.dot(
            self.grad_p, self.step_old
        ) * self.trust_radius_old

    def compute_trust_radius(self):
        """
        compute trust radius
        """
        ltest = self.e_wolfe and (
            self.nr_step_length_old > self.trust_radius_old + 1.0e-8
        )
        if ltest:
            a = 1.5
        else:
            a = 1.1

        if self.lwolfe:
            a = 2.0 * a
        self.trust_radius = min(
            self.trust_radius_max, a * self.trust_radius_old, self.nr_step_length
        )

        if self.trust_radius < self.trust_radius_min:
            # the history is reset
            if self.tr_min_hit == 1:
                self.log("history already reset at previous step: stopping")
                self.tr_min_hit = 2
            else:
                self.tr_min_hit = 1

            self.log("small trust_radius: resetting bfgs history")
            self.reset_bfgs()
            self.step = -np.dot(self.inv_hess, self.grad)
            self.nr_step_length = self.scnorm(self.step)
            self.step = self.step / self.nr_step_length
            self.trust_radius = min(self.trust_radius_min, self.nr_step_length)
        else:
            self.tr_min_hit = 0

    def scnorm(self, vec):
        """
        compute the norm of a vector with the metric
        """
        scnorm = 0.0
        for i in range(self.nat):
            ss = 0.0
            for k in range(3):
                for l in range(3):
                    ss = (
                        ss
                        + vec[k + i * 3]
                        * self.metric[k + i * 3, l + i * 3]
                        * vec[l + i * 3]
                    )
            scnorm = max(scnorm, np.sqrt(ss))
        return scnorm

    def terminate_bfgs(self):
        """
        terminate bfgs
        """
        if self.conv_bfgs:
            self.log("")
            self.log(
                f"bfgs converged in {self.scf_iter:3d} scf cycles and {self.bfgs_iter:3d} bfgs steps"
            )
            self.log(
                f"(criteria: energy < {self.energy_thr:8.2e} Ry, force < {self.grad_thr:8.2e} Ry/Bohr)"
            )
            self.log("")
            self.log("End of BFGS Geometry Optimization")
            self.log("")
            self.log(f"Final energy = {self.energy:18.10f} Ry")
        else:
            self.log("The maximum number of steps has been reached")

    def read_constants_positions_pwxml(self):
        """
        extract constants from pwscf.save/data-file-schema.xml
        """
        root_dir = os.getcwd() + "/"
        work_dir = root_dir + self.folder_name + str(self.scf_iter) + "/"
        xml_file = (
            work_dir + self.outdir + self.pw_prefix + ".save/data-file-schema.xml"
        )
        tree = ET.parse(xml_file)
        root = tree.getroot()

        if root.find("input/dft/hybrid"):
            self.l_exx = True

        # extract nat, alat, at and bg
        atomic_structure = root.find("input/atomic_structure")
        nat = int(atomic_structure.attrib["nat"])
        alat = float(atomic_structure.attrib["alat"])

        # extract at and bg
        at_str = []
        for i in [1, 2, 3]:
            aa = atomic_structure.find(f"cell/a{i}").text.split()
            at_str.append(aa)
        at = np.array(at_str, dtype=np.float64) / alat

        bg_str = []
        for i in [1, 2, 3]:
            bb = root.find(f"output/basis_set/reciprocal_lattice/b{i}").text.split()
            bg_str.append(bb)
        bg = np.array(bg_str, dtype=np.float64)

        # extract data for 'atomic species' and 'atomic coordinates'
        atoms = atomic_structure.findall("atomic_positions/atom")
        atoms_str = [atom.attrib["name"] for atom in atoms]
        pos_in_str = [atom.text.split() for atom in atoms]
        pos_in = np.array(pos_in_str, dtype=np.float64) / alat
        pos_in = np.ravel(pos_in)

        self.nat = nat
        self.alat = alat
        self.at = at
        self.bg = bg
        self.atoms = atoms_str
        self.pos_in = pos_in

    def read_ground_state_forces_energy_pwxml(self):
        """
        extract forces and energy from pwscf.save/data-file-schema.xml
        """
        root_dir = os.getcwd() + "/"
        work_dir = root_dir + self.folder_name + str(self.scf_iter) + "/"
        xml_file = (
            work_dir + self.outdir + self.pw_prefix + ".save/data-file-schema.xml"
        )

        tree = ET.parse(xml_file)
        root = tree.getroot()

        forces_str = root.find("output/forces").text.replace("\n", " ").split()
        forces = np.array(forces_str, dtype=np.float64).reshape((self.nat, 3))
        # convert Hartree/Bohr to Ry/Bohr
        forces = forces * Hartree / Rydberg

        gs_e = float(root.findall("output/total_energy/etot")[-1].text)
        # convert Hartree to Ry
        gs_e = gs_e * Hartree / Rydberg

        self.gs_e = gs_e
        self.force_g = forces

    def read_excited_state_forces_energy(self):
        """
        extract the excitation energy and the excited state forces from the wbse.json file
        """
        root_dir = os.getcwd() + "/"
        work_dir = root_dir + self.folder_name + str(self.scf_iter) + "/"
        json_file = work_dir + self.outdir + self.west_prefix + ".wbse.save/wbse.json"

        with open(json_file, "r") as f:
            data = json.load(f)

        # load excited state forces from wbse.json
        force_e = np.array(
            data["output"]["forces"]["forces_corrected"], dtype=np.float64
        )
        force_e = np.reshape(force_e, (self.nat, 3))

        # total forces are the sum of ground state forces and the excited state forces
        force = self.force_g + force_e
        # grad is the negative of the force
        grad_in = -force
        # change from cart to cry
        grad_in = grad_in * self.alat
        grad_in = np.dot(grad_in, self.at.T)
        self.grad_in = np.ravel(grad_in)

        # load the excitation energy
        ind_es = int(data["input"]["wbse_control"]["forces_state"])
        exc_e = np.array(data["exec"]["davitr"][-1]["ev"][ind_es - 1], dtype=np.float64)

        self.energy = self.gs_e + exc_e
